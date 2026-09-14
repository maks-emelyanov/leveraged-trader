from __future__ import annotations

import io
import json
import random
import unicodedata
import unittest
from datetime import UTC, datetime
from unittest.mock import call, patch
from urllib.parse import quote, unquote

import pandas as pd
from rich.console import Console

from leveraged_trader.output import (
    _DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH,
    _DIAGNOSTIC_NESTED_JSON_MAX_DEPTH,
    DEFAULT_NON_TERMINAL_WIDTH,
    STATUS_STYLES,
    TableColumn,
    WorkflowReporter,
    _decode_diagnostic_json_escape_unit,
    _is_nonzero_count,
    _redact_escaped_json_assignments,
    format_decimal_2,
    format_duration,
    format_int,
    format_message,
    safe_diagnostic_text,
)


class OutputTests(unittest.TestCase):
    def test_safe_diagnostic_text_bounds_string_and_byte_sources_before_rendering(self) -> None:
        class GuardedText(str):
            def __str__(self) -> str:
                raise AssertionError("string subclass __str__ must not be called")

            def __getitem__(self, key: object) -> str:
                raise AssertionError("string subclass __getitem__ must not be called")

        class GuardedBytes(bytes):
            def __str__(self) -> str:
                raise AssertionError("bytes subclass __str__ must not be called")

            def __getitem__(self, key: object) -> bytes:
                raise AssertionError("bytes subclass __getitem__ must not be called")

        class GuardedBytearray(bytearray):
            def __str__(self) -> str:
                raise AssertionError("bytearray subclass __str__ must not be called")

            def __getitem__(self, key: object) -> bytearray:
                raise AssertionError("bytearray subclass __getitem__ must not be called")

        sources = (
            GuardedText("token=SUPERSECRET123" + "A" * 10_000),
            GuardedBytes(b"token=SUPERSECRET123" + b"A" * 10_000),
            GuardedBytearray(b"token=SUPERSECRET123" + b"A" * 10_000),
            memoryview(b"token=SUPERSECRET123" + b"A" * 10_000),
        )
        for source in sources:
            with self.subTest(source_type=type(source).__name__):
                diagnostic = safe_diagnostic_text(source, max_chars=64)

                self.assertIn("[redacted credential]", diagnostic)
                self.assertNotIn("SUPERSECRET123", diagnostic)
                self.assertLessEqual(len(diagnostic), 64)

    def test_safe_diagnostic_text_uses_type_only_fallback_when_str_fails(self) -> None:
        class BrokenDiagnostic:
            def __str__(self) -> str:
                raise RuntimeError("TOPSECRET123")

        diagnostic = safe_diagnostic_text(BrokenDiagnostic())

        self.assertEqual(diagnostic, "<unprintable BrokenDiagnostic>")
        self.assertNotIn("TOPSECRET123", diagnostic)

    def test_safe_diagnostic_text_never_invokes_arbitrary_successful_str(self) -> None:
        calls: list[object] = []

        class ExpensiveDiagnostic:
            def __str__(self) -> str:
                calls.append(self)
                return "provider response " + "A" * 1_000_000

        diagnostic = safe_diagnostic_text(ExpensiveDiagnostic())

        self.assertEqual(diagnostic, "<unprintable ExpensiveDiagnostic>")
        self.assertEqual(calls, [])

    def test_safe_diagnostic_text_extracts_exception_args_without_invoking_render_hooks(self) -> None:
        exception_str_calls: list[object] = []
        payload_str_calls: list[object] = []

        class DiagnosticError(RuntimeError):
            def __str__(self) -> str:
                exception_str_calls.append(self)
                return "unsafe custom rendering"

        class DiagnosticPayload:
            def __str__(self) -> str:
                payload_str_calls.append(self)
                return "unsafe payload rendering"

        diagnostic = safe_diagnostic_text(DiagnosticError("provider returned token=TOPSECRET123; retry"))
        payload_diagnostic = safe_diagnostic_text(RuntimeError(DiagnosticPayload()))

        self.assertEqual(diagnostic, "provider returned [redacted credential]; retry")
        self.assertEqual(payload_diagnostic, "<unprintable DiagnosticPayload>")
        self.assertEqual(exception_str_calls, [])
        self.assertEqual(payload_str_calls, [])

    def test_safe_diagnostic_text_preserves_common_exact_scalars(self) -> None:
        cases = (
            (None, "None"),
            (True, "True"),
            (42, "42"),
            (1.25, "1.25"),
            (2 + 3j, "(2+3j)"),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(safe_diagnostic_text(value), expected)

    def test_safe_diagnostic_text_redacts_secrets_crossing_its_scan_boundary(self) -> None:
        scan_chars = 512 * 4
        cases = (
            ("Q", 1),
            ("YZ", 1),
            ("opaquecredential1234", 18),
            ("Q" * 512, 510),
        )
        for secret, retained_prefix_length in cases:
            with self.subTest(secret_length=len(secret)):
                diagnostic = safe_diagnostic_text(
                    "\x00" * (scan_chars - retained_prefix_length) + secret,
                    sensitive_values=(secret,),
                )

                self.assertIn("redacted", diagnostic)
                self.assertNotIn(secret[:retained_prefix_length], diagnostic)
                self.assertTrue(all(character.isprintable() for character in diagnostic))

    def test_safe_diagnostic_text_redacts_canonically_equivalent_sensitive_values(self) -> None:
        nfc_secret = "café-credential-7Q"
        nfd_secret = unicodedata.normalize("NFD", nfc_secret)
        mixed_secret = "café-o\u0308-credential-7Q"
        mixed_configured = unicodedata.normalize("NFC", mixed_secret)
        reordered_secret = "a\u0315\u0301\u0323-credential-7Q"
        reordered_configured = unicodedata.normalize("NFC", reordered_secret)

        cases = (
            (nfc_secret, nfd_secret),
            (nfd_secret, nfc_secret),
            (mixed_configured, mixed_secret),
            (reordered_configured, reordered_secret),
        )
        for configured, reflected in cases:
            with self.subTest(configured=repr(configured), reflected=repr(reflected)):
                self.assertEqual(
                    unicodedata.normalize("NFD", configured),
                    unicodedata.normalize("NFD", reflected),
                )
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {reflected}",
                    sensitive_values=(configured,),
                )

                self.assertEqual(diagnostic, "provider reflected [redacted]")
                self.assertNotIn(reflected, diagnostic)

    def test_safe_diagnostic_text_redacts_canonical_equivalence_across_encodings(self) -> None:
        reflected = "café-o\u0308-credential-7Q"
        configured = unicodedata.normalize("NFC", reflected)
        encoded_spellings = (
            json.dumps(reflected, ensure_ascii=True)[1:-1],
            quote(reflected, safe=""),
            quote(json.dumps(reflected, ensure_ascii=True)[1:-1], safe=""),
        )

        for encoded in encoded_spellings:
            with self.subTest(encoded=encoded):
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {encoded}",
                    sensitive_values=(configured,),
                )

                self.assertEqual(diagnostic, "provider reflected [redacted]")
                self.assertNotIn(encoded, diagnostic)

    def test_safe_diagnostic_text_redacts_normalization_expansion_across_scan_boundary(self) -> None:
        max_chars = 64
        secret = "é" * max_chars
        expanded_secret = unicodedata.normalize("NFD", secret)
        retained_prefix_length = max_chars

        diagnostic = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - retained_prefix_length) + expanded_secret,
            max_chars=max_chars,
            sensitive_values=(secret,),
        )

        self.assertEqual(diagnostic, "[redacted]")
        self.assertFalse(expanded_secret.startswith(diagnostic))
        self.assertLessEqual(len(diagnostic), max_chars)

    def test_safe_diagnostic_text_does_not_disclose_an_oversized_secret_prefix(self) -> None:
        for length in (513, 3_000):
            with self.subTest(length=length):
                secret = "Q" * length

                diagnostic = safe_diagnostic_text(
                    f"provider echoed {secret}",
                    sensitive_values=(secret,),
                )

                self.assertEqual(diagnostic, "[redacted credential]")
                self.assertNotIn(secret[:512], diagnostic)

    def test_safe_diagnostic_text_bounds_oversized_secret_marker_at_every_small_limit(self) -> None:
        for max_chars in range(1, len("[redacted credential]")):
            with self.subTest(max_chars=max_chars):
                secret = "§" * (max_chars + 1)

                diagnostic = safe_diagnostic_text(
                    f"provider echoed {secret}",
                    max_chars=max_chars,
                    sensitive_values=(secret,),
                )

                self.assertLessEqual(len(diagnostic), max_chars)
                self.assertNotIn("§", diagnostic)
                self.assertTrue(all(character.isprintable() for character in diagnostic))

    def test_safe_diagnostic_text_never_introduces_a_sensitive_value_through_its_marker(self) -> None:
        for sensitive_value in ("credential", "redacted", "[", "]"):
            with self.subTest(sensitive_value=sensitive_value):
                diagnostic = safe_diagnostic_text(
                    "api_key=provider-value",
                    sensitive_values=(sensitive_value,),
                )

                self.assertNotIn(sensitive_value, diagnostic)
                self.assertTrue(all(character.isprintable() for character in diagnostic))

    def test_oversized_secret_fallback_scans_later_values_for_marker_collisions(self) -> None:
        diagnostic = safe_diagnostic_text(
            "provider failure",
            sensitive_values=("X" * 1_000, "credential"),
        )

        self.assertNotIn("credential", diagnostic)
        self.assertLessEqual(len(diagnostic), 512)
        self.assertTrue(all(character.isprintable() for character in diagnostic))

    def test_oversized_whitespace_padded_secret_cannot_shrink_into_fallback_marker(self) -> None:
        secret = "      credential      "

        diagnostic = safe_diagnostic_text(
            "provider failure",
            max_chars=len("[redacted credential]"),
            sensitive_values=(secret,),
        )

        self.assertNotIn(secret.strip(), diagnostic)
        self.assertEqual(diagnostic, "")

    def test_safe_diagnostic_text_redacts_repr_escaped_sensitive_values(self) -> None:
        secrets = ("opaque\\credential123", "line-one\nline-two")

        for secret in secrets:
            with self.subTest(secret=repr(secret)):
                escaped_secret = repr(secret)
                diagnostic = safe_diagnostic_text(
                    f"provider raised RuntimeError({escaped_secret})",
                    sensitive_values=(secret,),
                )

                self.assertIn("redacted", diagnostic)
                self.assertNotIn(escaped_secret, diagnostic)
                self.assertNotIn(escaped_secret[1:-1], diagnostic)
                self.assertTrue(all(character.isprintable() for character in diagnostic))

                unquoted_diagnostic = safe_diagnostic_text(
                    f"provider escaped {escaped_secret[1:-1]}",
                    sensitive_values=(secret,),
                )
                self.assertIn("redacted", unquoted_diagnostic)
                self.assertNotIn(escaped_secret[1:-1], unquoted_diagnostic)

    def test_safe_diagnostic_text_redacts_equivalent_json_escape_spellings(self) -> None:
        secret = '§"/😀'
        escaped_spellings = (
            r"\u00A7\u0022\/\uD83D\uDE00",
            r"\u00a7\"\u002f\ud83d\ude00",
            r"§\u0022\/\uD83d\uDe00",
            json.dumps(secret, ensure_ascii=True)[1:-1],
        )

        for escaped_secret in escaped_spellings:
            with self.subTest(escaped_secret=escaped_secret):
                self.assertEqual(json.loads(f'"{escaped_secret}"'), secret)
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {escaped_secret}",
                    sensitive_values=(secret,),
                )

                self.assertEqual(diagnostic, "provider reflected [redacted]")

    def test_safe_diagnostic_text_redacts_equivalent_json_escapes_across_scan_boundary(self) -> None:
        secret = '§"/😀'
        escaped_spellings = (
            r"\u00A7\u0022\/\uD83D\uDE00",
            r"\u00a7\"\u002f\ud83d\ude00",
            r"§\u0022\/\uD83d\uDe00",
        )
        max_chars = 64

        for escaped_secret in escaped_spellings:
            with self.subTest(escaped_secret=escaped_secret):
                boundary_diagnostic = safe_diagnostic_text(
                    " " * (max_chars * 4 - 1) + escaped_secret,
                    max_chars=max_chars,
                    sensitive_values=(secret,),
                )

                self.assertEqual(boundary_diagnostic, "[redacted]")
                self.assertLessEqual(len(boundary_diagnostic), max_chars)

    def test_safe_diagnostic_text_redacts_url_percent_encoded_sensitive_values(self) -> None:
        cases = (
            (
                "A/b+§%Z",
                (
                    "%41%2F%62%2B%C2%A7%25%5A",
                    "%41%2f%62%2b%c2%a7%25%5a",
                    "A%2fb%2B%C2%a7%25Z",
                ),
            ),
            ("space separated", ("space+separated", "%73pace%20separated")),
        )

        for secret, encoded_spellings in cases:
            for encoded in encoded_spellings:
                with self.subTest(secret=secret, encoded=encoded):
                    diagnostic = safe_diagnostic_text(
                        f"provider reflected {encoded}",
                        sensitive_values=(secret,),
                    )

                    self.assertEqual(diagnostic, "provider reflected [redacted]")
                    self.assertNotIn(encoded, diagnostic)

    def test_safe_diagnostic_text_redacts_composed_sensitive_value_encodings(self) -> None:
        secret = "S3cr/et+§%Z"
        json_spelling = json.dumps(secret, ensure_ascii=True)[1:-1]
        encoded_spellings = (
            quote(quote(secret, safe=""), safe=""),
            quote(json_spelling, safe=""),
            json.dumps(quote(secret, safe=""), ensure_ascii=True)[1:-1],
            json.dumps(json_spelling, ensure_ascii=True)[1:-1],
        )

        for encoded in encoded_spellings:
            with self.subTest(encoded=encoded):
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {encoded}",
                    sensitive_values=(secret,),
                )

                self.assertEqual(diagnostic, "provider reflected [redacted]")
                self.assertNotIn(encoded, diagnostic)

    def test_safe_diagnostic_text_fails_closed_beyond_composed_sensitive_value_depth(self) -> None:
        secret = "S3cr/et+§%Z"

        for depth in (3, _DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH + 1, 12):
            encoded = secret
            for _ in range(depth):
                encoded = quote(encoded, safe="")

            with self.subTest(depth=depth):
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {encoded}",
                    sensitive_values=(secret,),
                )

                self.assertIn("[redacted", diagnostic)
                self.assertNotIn(encoded, diagnostic)
                decoded = diagnostic
                for _ in range(depth):
                    decoded = unquote(decoded)
                self.assertNotIn(secret, decoded)

    def test_safe_diagnostic_text_redacts_composed_encoding_across_scan_boundary(self) -> None:
        secret = "S3cr/et+§%Z"
        encoded = quote(quote(secret, safe=""), safe="")
        max_chars = 64

        diagnostic = safe_diagnostic_text(
            " " * (max_chars * 4 - 1) + encoded,
            max_chars=max_chars,
            sensitive_values=(secret,),
        )

        self.assertEqual(diagnostic, "[redacted]")
        self.assertLessEqual(len(diagnostic), max_chars)
        self.assertNotIn(encoded[:1], diagnostic)

    def test_safe_diagnostic_text_fails_closed_for_deep_encoding_across_scan_boundary(self) -> None:
        secret = "ABC/def"
        encoded = secret
        for _ in range(1_200):
            encoded = quote(encoded, safe="")
        max_chars = 64

        diagnostic = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - 3) + encoded,
            max_chars=max_chars,
            sensitive_values=(secret,),
        )

        self.assertIn("[redacted", diagnostic)
        self.assertNotIn(secret[:3], diagnostic)
        self.assertLessEqual(len(diagnostic), max_chars)

    def test_safe_diagnostic_text_redacts_fully_escaped_composition_across_scan_boundary(self) -> None:
        def unicode_escape_layer(value: str) -> str:
            return "".join(f"\\u{ord(character):04X}" for character in value)

        secret = "AB" + "C" * 62
        encoded = secret
        for _ in range(_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH):
            encoded = unicode_escape_layer(encoded)
        retained_encoded_chars = len(secret[:2]) * (6**_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH)
        max_chars = 512

        diagnostic = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - retained_encoded_chars) + encoded,
            max_chars=max_chars,
            sensitive_values=(secret,),
        )
        decoded_prefix = diagnostic
        for _ in range(_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH):
            decoded_prefix = json.loads(f'"{decoded_prefix}"')

        self.assertIn("[redacted", diagnostic)
        self.assertFalse(secret.startswith(decoded_prefix))
        self.assertLessEqual(len(diagnostic), max_chars)

    def test_safe_diagnostic_text_preserves_permissive_mixed_percent_spellings(self) -> None:
        cases = (
            ("+A%Z", "+%41%25Z"),
            ("%41+", "%25%341+"),
        )

        for secret, encoded in cases:
            with self.subTest(secret=secret, encoded=encoded):
                diagnostic = safe_diagnostic_text(
                    f"provider reflected {encoded}",
                    sensitive_values=(secret,),
                )

                self.assertEqual(diagnostic, "provider reflected [redacted]")
                self.assertNotIn(encoded, diagnostic)

    def test_safe_diagnostic_text_redacts_percent_encoding_across_scan_boundary(self) -> None:
        secret = "A/b+§%Z"
        encoded = "%41%2f%62%2B%C2%a7%25%5a"
        max_chars = 64

        for retained_prefix_length in (1, 2, 3, len(encoded) - 1):
            with self.subTest(retained_prefix_length=retained_prefix_length):
                diagnostic = safe_diagnostic_text(
                    "\x00" * (max_chars * 4 - retained_prefix_length) + encoded,
                    max_chars=max_chars,
                    sensitive_values=(secret,),
                )

                self.assertEqual(diagnostic, "[redacted]")
                self.assertLessEqual(len(diagnostic), max_chars)
                self.assertNotIn(encoded[:retained_prefix_length], diagnostic)

    def test_safe_diagnostic_text_bounds_work_for_long_encoded_secret_prefixes(self) -> None:
        max_chars = 4_096
        secret = "A" * (max_chars - 1) + "B"

        unmatched = safe_diagnostic_text(
            "A" * (max_chars * 10),
            max_chars=max_chars,
            sensitive_values=(secret,),
        )
        percent_encoded = safe_diagnostic_text(
            "A" * (max_chars - 1) + "%42",
            max_chars=max_chars,
            sensitive_values=(secret,),
        )
        json_encoded = safe_diagnostic_text(
            "A" * (max_chars - 1) + r"\u0042",
            max_chars=max_chars,
            sensitive_values=(secret,),
        )

        self.assertEqual(len(unmatched), max_chars)
        self.assertTrue(unmatched.endswith("…"))
        self.assertEqual(percent_encoded, "[redacted]")
        self.assertEqual(json_encoded, "[redacted]")

        ambiguous_max_chars = 16_384
        ambiguous_secret = "A" * 126 + "+B"
        unmatched_ambiguous = safe_diagnostic_text(
            "A" * (ambiguous_max_chars * 4),
            max_chars=ambiguous_max_chars,
            sensitive_values=(ambiguous_secret,),
        )
        self.assertEqual(len(unmatched_ambiguous), ambiguous_max_chars)
        self.assertTrue(unmatched_ambiguous.endswith("…"))

    def test_safe_diagnostic_text_redacts_randomized_mixed_percent_spellings(self) -> None:
        generator = random.Random(0xC0DEC)
        alphabet = tuple("AbZ09-/+% ") + ("§", "é", "中", "😀")

        def randomized_percent_spelling(value: str) -> str:
            pieces: list[str] = []
            for character in value:
                encoded = "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
                if character == " ":
                    choices = ["+", encoded.lower(), encoded.upper()]
                elif character in {"+", "%"}:
                    choices = [encoded.lower(), encoded.upper()]
                else:
                    choices = [character, encoded.lower(), encoded.upper()]
                pieces.append(generator.choice(choices))
            return "".join(pieces)

        for case_number in range(200):
            secret = "S" + "".join(generator.choices(alphabet, k=generator.randint(1, 20))) + "E"
            encoded_secret = randomized_percent_spelling(secret)
            with self.subTest(case_number=case_number, encoded_secret=encoded_secret):
                self.assertEqual(
                    safe_diagnostic_text(
                        f"provider reflected {encoded_secret}",
                        sensitive_values=(secret,),
                    ),
                    "provider reflected [redacted]",
                )

    def test_safe_diagnostic_text_redacts_randomized_valid_json_escape_spellings(self) -> None:
        generator = random.Random(0x5AFE)
        alphabet = tuple("AbZ09-/\\\"'") + (
            "§",
            "é",
            "中",
            "😀",
            "\b",
            "\f",
            "\n",
            "\r",
            "\t",
        )
        simple_escapes = {
            '"': r"\"",
            "\\": r"\\",
            "/": r"\/",
            "\b": r"\b",
            "\f": r"\f",
            "\n": r"\n",
            "\r": r"\r",
            "\t": r"\t",
        }

        def hex_escape(code_unit: int) -> str:
            digits = f"{code_unit:04x}"
            randomized_digits = "".join(
                character.upper() if generator.getrandbits(1) else character for character in digits
            )
            return rf"\u{randomized_digits}"

        def randomized_json_spelling(value: str) -> str:
            pieces = []
            for character in value:
                code_point = ord(character)
                if code_point > 0xFFFF:
                    offset = code_point - 0x10000
                    unicode_escape = hex_escape(0xD800 + (offset >> 10)) + hex_escape(0xDC00 + (offset & 0x3FF))
                else:
                    unicode_escape = hex_escape(code_point)
                choices = [unicode_escape]
                if character in simple_escapes:
                    choices.append(simple_escapes[character])
                if code_point >= 0x20 and character not in {'"', "\\"}:
                    choices.append(character)
                pieces.append(generator.choice(choices))
            return "".join(pieces)

        for case_number in range(200):
            secret = "S" + "".join(generator.choices(alphabet, k=generator.randint(1, 20))) + "E"
            escaped_secret = randomized_json_spelling(secret)
            with self.subTest(case_number=case_number, escaped_secret=escaped_secret):
                self.assertEqual(json.loads(f'"{escaped_secret}"'), secret)
                self.assertEqual(
                    safe_diagnostic_text(
                        f"provider reflected {escaped_secret}",
                        sensitive_values=(secret,),
                    ),
                    "provider reflected [redacted]",
                )

    def test_safe_diagnostic_text_redacts_quoted_credential_assignments(self) -> None:
        cases = (
            ("provider returned token=SUPERSECRET123; retry later", "SUPERSECRET123"),
            ("provider returned authorization is Bearer SUPERSECRET123", "SUPERSECRET123"),
            ("provider returned Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
            ("provider returned Authorization=Token SUPERSECRET123", "SUPERSECRET123"),
            (
                "provider returned Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/request",
                "Credential=AKIAEXAMPLE/request",
            ),
            ('provider returned {"token":"SUPERSECRET123"}', "SUPERSECRET123"),
            (
                'provider returned {"api_key": "secret with spaces, commas; and separators"}',
                "secret with spaces, commas; and separators",
            ),
            ("provider returned {'api_secret': 'SUPERSECRET123'}", "SUPERSECRET123"),
            ('provider returned {"api_key_id":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"api_secret_key":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"access_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"refresh_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"session_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"auth_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"id_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"client_secret":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"secret_access_key":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"github_access_token":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"aws_secret_access_key":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"openai_api_key":"SUPERSECRET123"}', "SUPERSECRET123"),
            ('provider returned {"password":"SUPERSECRET123"}', "SUPERSECRET123"),
            (
                'provider returned {"credential":"escaped \\"quoted\\" secret","status":"denied"}',
                'escaped \\"quoted\\" secret',
            ),
        )

        for diagnostic, secret in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                self.assertNotIn(secret, redacted)

        adjacent_field = safe_diagnostic_text(
            '{"token":"secret, with spaces; through the closing quote","status":"denied"}'
        )
        self.assertNotIn("through the closing quote", adjacent_field)
        self.assertIn('"status":"denied"', adjacent_field)

    def test_safe_diagnostic_text_redacts_semantic_json_keys_with_valid_escapes(self) -> None:
        cases = (
            r'{"access\u005ftoken":"ACCESS-SECRET","status":"denied"}',
            r'{"\u006fpenai_api_key":"API-SECRET","status":"denied"}',
            r'{"aws_secret\u005faccess\u005fkey":"AWS-SECRET","status":"denied"}',
        )

        for diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                self.assertNotIn("SECRET", redacted)
                self.assertIn('"status":"denied"', redacted)

        invalid_escape = r'{"access\x5ftoken":"ordinary","status":"ok"}'
        self.assertEqual(safe_diagnostic_text(invalid_escape), invalid_escape)

    def test_safe_diagnostic_text_redacts_dot_separated_credential_keys(self) -> None:
        cases = (
            ("provider returned api.key=PLAINVALUE123456789; retry later", "PLAINVALUE123456789"),
            (
                "provider returned client.secret.value='CLIENTVALUE123456789'; retry later",
                "CLIENTVALUE123456789",
            ),
            ('{"api.key":"JSONVALUE123456789","status":"denied"}', "JSONVALUE123456789"),
            ('{"aws.secret.access.key":"AWSVALUE123456789","status":"denied"}', "AWSVALUE123456789"),
            (r'{"api\u002ekey":"ESCAPEDVALUE123456789","status":"denied"}', "ESCAPEDVALUE123456789"),
        )

        for diagnostic, secret in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                self.assertNotIn(secret, redacted)

    def test_format_message_redacts_overlong_ambiguous_json_credential_key(self) -> None:
        diagnostic = '{"' + "A" * 510 + r'\u005faccess\u005ftoken":"EXPOSED-VALUE-42","status":"denied"}'

        redacted = format_message(diagnostic)

        self.assertIn("[redacted credential]", redacted)
        self.assertNotIn("EXPOSED-VALUE-42", redacted)

    def test_safe_diagnostic_text_redacts_complete_parameterized_authorization_values(self) -> None:
        cases = (
            (
                'Authorization: Digest username="operator", realm="broker", nonce="NONCE", '
                'uri="/orders;settled=true", response="DIGEST-RESPONSE", qop=auth\nstatus=401',
                ("NONCE", "DIGEST-RESPONSE", "response="),
            ),
            (
                "Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/request, "
                "SignedHeaders=host;x-amz-date, Signature=AWS-SIGNATURE; retry later",
                ("AKIAEXAMPLE", "AWS-SIGNATURE", "Signature="),
            ),
            (
                'Authorization: Digest username=operator, nonce=NONCE,\r\n response="FOLDED-RESPONSE"\nstatus=403',
                ("NONCE", "FOLDED-RESPONSE", "response="),
            ),
        )

        for diagnostic, secrets in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                for secret in secrets:
                    self.assertNotIn(secret, redacted)

        self.assertIn("status=401", safe_diagnostic_text(cases[0][0]))
        self.assertIn("status=403", safe_diagnostic_text(cases[2][0]))

    def test_safe_diagnostic_text_redacts_credential_label_suffixes(self) -> None:
        cases = (
            (
                'Authorization header: Digest username="operator", nonce="NONCE", '
                'response="DIGEST-RESPONSE"\nstatus=401',
                ("NONCE", "DIGEST-RESPONSE", "response="),
            ),
            (
                "authorization_header=AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/request, "
                "SignedHeaders=host;x-amz-date, Signature=AWS-SIGNATURE\nstatus=403",
                ("AKIAEXAMPLE", "AWS-SIGNATURE", "Signature="),
            ),
            (
                "authorizationHeader=AWS4-HMAC-SHA256 Credential=AKIACAMEL/request, "
                "SignedHeaders=host;x-amz-date, Signature=CAMEL-SIGNATURE\nstatus=403",
                ("AKIACAMEL", "CAMEL-SIGNATURE", "Signature="),
            ),
            ("api_key_value=API-SECRET; retry later", ("API-SECRET",)),
            ("apiKeyValue=CAMEL-API-SECRET; retry later", ("CAMEL-API-SECRET",)),
            ('client_secret_value: "CLIENT SECRET WITH SPACES"; retry later', ("CLIENT SECRET",)),
            ('clientSecretValue: "CAMEL CLIENT SECRET"; retry later', ("CAMEL CLIENT",)),
            ('{"api_key_value":"JSON-SECRET","status":"denied"}', ("JSON-SECRET",)),
            ('{"apiKeyValue":"CAMEL-JSON-SECRET","status":"denied"}', ("CAMEL-JSON-SECRET",)),
        )

        for diagnostic, secrets in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                for secret in secrets:
                    self.assertNotIn(secret, redacted)

        self.assertIn("status=401", safe_diagnostic_text(cases[0][0]))
        self.assertIn("status=403", safe_diagnostic_text(cases[1][0]))

    def test_safe_diagnostic_text_redacts_vendor_prefixed_camel_case_credentials(self) -> None:
        assignments = (
            "awsSecretAccessKeyValue=VERYSECRETVALUE; retry later",
            "AwsSecretAccessKeyValue=VERYSECRETVALUE; retry later",
            "AWSSecretAccessKeyValue=VERYSECRETVALUE; retry later",
            "githubAccessTokenValue=GITHUB-SECRET; retry later",
            "GitHubAccessTokenValue=GITHUB-SECRET; retry later",
            "OpenAIAPIKeyValue=OPENAI-SECRET; retry later",
            "MyVendorClientSecretValue=CLIENT-SECRET; retry later",
            "googleCloudAccessTokenValue=GOOGLE-SECRET; retry later",
            "vendorClientSecretValue=CLIENT-SECRET; retry later",
            "xApiKey=X-API-SECRET; retry later",
            "apiToken=API-TOKEN-SECRET; retry later",
            "vendorAuthorizationHeader=Basic BASICSECRET\nstatus=401",
            "authorizationHeaderValue=Bearer BEARERSECRET\nstatus=403",
        )

        for diagnostic in assignments:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted", redacted)
                self.assertNotIn("SECRET", redacted)

        semantic_json = (
            '{"openaiApiKeyValue":"sk-JSON-SECRET","status":"denied"}',
            '{"OpenAIAPIKeyValue":"sk-JSON-SECRET","status":"denied"}',
            '{"githubAccessTokenValue":"ghp-JSON-SECRET","status":"denied"}',
            '{"GitHubAccessTokenValue":"ghp-JSON-SECRET","status":"denied"}',
            '{"AwsSecretAccessKeyValue":"AWS-JSON-SECRET","status":"denied"}',
            '{"MyVendorClientSecretValue":"CLIENT-JSON-SECRET","status":"denied"}',
            '{"googleCloudAccessTokenValue":"GOOGLE-JSON-SECRET","status":"denied"}',
            '{"vendorClientSecretValue":"CLIENT-JSON-SECRET","status":"denied"}',
            '{"xApiKey":"X-JSON-SECRET","status":"denied"}',
            '{"apiToken":"TOKEN-JSON-SECRET","status":"denied"}',
            '{"authorizationHeaderValue":"Basic JSON-SECRET","status":"denied"}',
        )

        for diagnostic in semantic_json:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)

                self.assertIn("[redacted credential]", redacted)
                self.assertNotIn("SECRET", redacted)
                self.assertIn('"status":"denied"', redacted)

    def test_safe_diagnostic_text_preserves_ordinary_credential_label_prose(self) -> None:
        diagnostic = "Authorization header is missing; the API key value is absent; the client secret value is unset."

        self.assertEqual(safe_diagnostic_text(diagnostic), diagnostic)

    def test_safe_diagnostic_text_redacts_url_userinfo_without_hiding_destination(self) -> None:
        cases = (
            (
                "GET https://operator:TOPSECRET123@provider.example/path failed",
                "GET https://operator:[redacted]@provider.example/path failed",
            ),
            (
                "GET https://operator:TOP%53ECRET%40123@provider.example/path failed",
                "GET https://operator:[redacted]@provider.example/path failed",
            ),
            (
                "GET https://operator:TOP!$&'()*+,;=SECRET@provider.example:8443/path failed",
                "GET https://operator:[redacted]@provider.example:8443/path failed",
            ),
            (
                "GET https://operator:TOP SECRET@provider.example/path failed",
                "GET https://operator:[redacted]@provider.example/path failed",
            ),
            (
                "GET https://operator:TOP VERY SECRET@provider.example/path failed",
                "GET https://operator:[redacted]@provider.example/path failed",
            ),
            (
                "GET https://operator:pass@@provider.example/path failed",
                "GET https://operator:[redacted]@provider.example/path failed",
            ),
            (
                "GET https://operator@provider.example/path failed",
                "GET https://[redacted]@provider.example/path failed",
            ),
            (
                "proxy socks5://operator:TOPSECRET123@proxy.example failed",
                "proxy socks5://operator:[redacted]@proxy.example failed",
            ),
            (
                "database postgresql://operator:TOPSECRET123@db.example/trades failed",
                "database postgresql://operator:[redacted]@db.example/trades failed",
            ),
            (
                "socket WSS://operator:TOPSECRET123@provider.example/feed failed",
                "socket WSS://operator:[redacted]@provider.example/feed failed",
            ),
            (
                "nested jdbc:postgresql://operator:TOPSECRET123@db.example/trades failed",
                "nested jdbc:postgresql://operator:[redacted]@db.example/trades failed",
            ),
        )

        for diagnostic, expected in cases:
            with self.subTest(diagnostic=diagnostic):
                self.assertEqual(safe_diagnostic_text(diagnostic), expected)

        ordinary_url = "GET https://provider.example:443/path failed"
        self.assertEqual(safe_diagnostic_text(ordinary_url), ordinary_url)
        ordinary_proxy = "proxy socks5://proxy.example:1080 failed"
        self.assertEqual(safe_diagnostic_text(ordinary_proxy), ordinary_proxy)

        pathless_urls = "URLs ftp://host and ssh://operator:TOPSECRET123@other.example"
        self.assertEqual(
            safe_diagnostic_text(pathless_urls),
            "URLs ftp://host and ssh://operator:[redacted]@other.example",
        )

        pathless_url_and_email = '{"url":"ftp://host","email":"user@example.com"}'
        redacted = safe_diagnostic_text(pathless_url_and_email)
        self.assertEqual(json.loads(redacted), {"url": "ftp://host", "email": "user@example.com"})

        credentialed_pathless_url_and_email = '{"url":"ftp://operator:TOPSECRET123@host","email":"user@example.com"}'
        redacted = safe_diagnostic_text(credentialed_pathless_url_and_email)
        self.assertEqual(
            json.loads(redacted),
            {"url": "ftp://operator:[redacted]@host", "email": "user@example.com"},
        )

    def test_safe_diagnostic_text_redacts_empty_password_userinfo_across_encodings(self) -> None:
        raw_url = "https://opaqueABC123xyz:@host.example/path"
        expected_url = "https://[redacted]:@host.example/path"

        self.assertEqual(safe_diagnostic_text(raw_url), expected_url)
        self.assertEqual(safe_diagnostic_text("https://:@host.example/path"), "https://:@host.example/path")
        self.assertEqual(
            safe_diagnostic_text("https://:TOPSECRET123@host.example/path"),
            "https://:[redacted]@host.example/path",
        )

        escaped_json = r'{"url":"https:\/\/opaqueABC123xyz:@host.example\/path","status":"failed"}'
        redacted_json = safe_diagnostic_text(escaped_json)
        self.assertEqual(
            json.loads(redacted_json),
            {"url": expected_url, "status": "failed"},
        )

        percent_encoded = quote(raw_url, safe="")
        self.assertEqual(unquote(safe_diagnostic_text(percent_encoded)), expected_url)

        composed = quote(
            r"https\u003A\u002F\u002FopaqueABC123xyz\u003A\u0040host.example\u002Fpath",
            safe="",
        )
        composed_redacted = unquote(safe_diagnostic_text(composed))
        self.assertIn("[redacted]", composed_redacted)
        self.assertNotIn("opaqueABC123xyz", composed_redacted)

        max_chars = 64
        visible_url = "https://opaqueABC"
        boundary_redacted = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - len(visible_url)) + raw_url,
            max_chars=max_chars,
        )
        self.assertEqual(boundary_redacted, "https://[redacted]")
        self.assertNotIn("opaqueABC", boundary_redacted)
        self.assertLessEqual(len(boundary_redacted), max_chars)

    def test_safe_diagnostic_text_redacts_percent_encoded_credential_structures(self) -> None:
        cases = (
            "api_key=TOPSECRET123; retry later",
            "Authorization: Bearer TOPSECRET123\nstatus=401",
            "Bearer TOPSECRET123; retry later",
            '{"access_token":"TOPSECRET123","status":"denied"}',
        )

        for raw in cases:
            encoded = quote(raw, safe="")
            with self.subTest(raw=raw):
                decoded = unquote(safe_diagnostic_text(encoded))

                self.assertIn("[redacted", decoded)
                self.assertNotIn("TOPSECRET123", decoded)

        raw_url = "GET https://operator:TOPSECRET123@provider.example/path failed"
        decoded_url = unquote(safe_diagnostic_text(quote(raw_url, safe="")))
        self.assertEqual(
            decoded_url,
            "GET https://operator:[redacted]@provider.example/path failed",
        )

        benign = quote("GET https://provider.example/path failed", safe="")
        self.assertEqual(safe_diagnostic_text(benign), benign)
        opaque_token = "api%2Dkey%2DABCDEF123456"
        self.assertIn("[redacted", safe_diagnostic_text(opaque_token))
        benign_token = "token%2Dbucket%2Dfilter%2Dv2beta3"
        self.assertEqual(safe_diagnostic_text(benign_token), benign_token)

    def test_safe_diagnostic_text_redacts_composed_percent_json_credential_structure(self) -> None:
        escaped_url = (
            r"GET \u0068ttps\u003A\u002F\u002Foperator\u003A"
            r"TOPSECRET123\u0040provider.example\u002Fpath failed"
        )
        encoded = quote(escaped_url, safe="")

        decoded = unquote(safe_diagnostic_text(encoded))

        self.assertIn("operator", decoded)
        self.assertIn("[redacted]", decoded)
        self.assertIn("provider.example", decoded)
        self.assertNotIn("TOPSECRET123", decoded)

    def test_safe_diagnostic_text_redacts_percent_encoded_userinfo_across_scan_boundary(self) -> None:
        raw = "GET https://operator:TOPSECRET123@provider.example/path failed"
        encoded = quote(raw, safe="")
        visible_chars = encoded.index("TOP") + len("TOP")
        max_chars = 64

        diagnostic = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - visible_chars) + encoded,
            max_chars=max_chars,
        )

        self.assertIn("[redacted]", diagnostic)
        self.assertNotIn("TOP", diagnostic)
        self.assertLessEqual(len(diagnostic), max_chars)

    def test_safe_diagnostic_text_redacts_json_escaped_assignment_at_scan_boundary(self) -> None:
        secret = "S" * 70
        encoded = "".join(f"\\u{ord(character):04x}" for character in f"token={secret}")
        max_chars = 512

        diagnostic = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - len(encoded)) + encoded,
            max_chars=max_chars,
        )
        decoded = json.loads(f'"{diagnostic}"')

        self.assertIn("[redacted", diagnostic)
        self.assertNotIn(secret, decoded)

    def test_safe_diagnostic_text_rescans_credentials_created_by_normalization(self) -> None:
        secret = "ABC DEF"
        for separator in ("\x00", "\t", "\n"):
            with self.subTest(separator=repr(separator)):
                diagnostic = safe_diagnostic_text(
                    f"ABC{separator}DEF",
                    sensitive_values=(secret,),
                )

                self.assertIn("[redacted", diagnostic)
                self.assertNotIn(secret, diagnostic)

                max_chars = 64
                boundary_diagnostic = safe_diagnostic_text(
                    "\x00" * (max_chars * 4 - 3) + f"ABC{separator}DEF",
                    max_chars=max_chars,
                    sensitive_values=(secret,),
                )
                self.assertIn("[redacted", boundary_diagnostic)
                self.assertNotIn(secret[:3], boundary_diagnostic)

        url = safe_diagnostic_text("https://user:TOPSECRET\n@host.example/path")
        self.assertEqual(url, "https://user:[redacted]@host.example/path")

        max_chars = 64
        visible_url = "https://user:TOP"
        boundary_url = safe_diagnostic_text(
            "\x00" * (max_chars * 4 - len(visible_url)) + "https://user:TOPSECRET\n@host.example/path",
            max_chars=max_chars,
        )
        self.assertEqual(boundary_url, "https://user:[redacted]")

        for separator in ("\n", "\r"):
            with self.subTest(authorization_separator=repr(separator)):
                authorization = f"Authorization{separator}:{separator}Bearer{separator}TOPSECRET123"
                self.assertEqual(safe_diagnostic_text(authorization), "[redacted]")

                visible_authorization = f"Authorization{separator}:{separator}Bearer{separator}TOP"
                boundary_authorization = safe_diagnostic_text(
                    "\x00" * (max_chars * 4 - len(visible_authorization)) + authorization,
                    max_chars=max_chars,
                )
                self.assertEqual(boundary_authorization, "[redacted]")

        url_prefix = "https://user:TOP\n"
        authorization = "Authorization: Bearer OTHERSECRET\n"
        url_suffix = "@host.example/path"
        for visible_authorization_chars in (1, 5, 15):
            with self.subTest(visible_authorization_chars=visible_authorization_chars):
                overlapping_url = safe_diagnostic_text(
                    "\x00" * (max_chars * 4 - len(url_prefix) - visible_authorization_chars)
                    + url_prefix
                    + authorization
                    + url_suffix,
                    max_chars=max_chars,
                )

                self.assertIn("[redacted]", overlapping_url)
                self.assertNotIn("TOP", overlapping_url)

    def test_safe_diagnostic_text_rescans_configured_secrets_after_composed_decoding(self) -> None:
        secret = "ABC DEF"
        encoded_separators = (
            r"\u0000",
            r"\u0009",
            r"\u000a",
            r"\u000d",
            "%00",
            "%09",
            "%0A",
            "%0D",
        )

        for encoded_separator in encoded_separators:
            encoded = f"ABC{encoded_separator}DEF"
            for diagnostic in (encoded, quote(encoded, safe="")):
                with self.subTest(encoded_separator=encoded_separator, diagnostic=diagnostic):
                    redacted = safe_diagnostic_text(
                        diagnostic,
                        sensitive_values=(secret,),
                    )

                    self.assertEqual(redacted, "[redacted]")

                    max_chars = 64
                    boundary_redacted = safe_diagnostic_text(
                        "\x00" * (max_chars * 4 - len("ABC")) + diagnostic,
                        max_chars=max_chars,
                        sensitive_values=(secret,),
                    )
                    self.assertEqual(boundary_redacted, "[redacted]")

    def test_safe_diagnostic_text_rescans_structures_after_composed_decoding_normalization(self) -> None:
        diagnostics = (
            r"Bearer\u0000TOPSECRET123",
            "Bearer%00TOPSECRET123",
            r"token\u0000=TOPSECRET123",
            "token%00%3DTOPSECRET123",
            r"Authorization\u0000:\u0009Bearer\u000aTOPSECRET123",
            "Authorization%00%3A%09Bearer%0ATOPSECRET123",
            (
                r"\u0068ttps\u003a\u002f\u002foperator\u003aTOPSECRET123"
                r"\u000a\u0040host.example\u002fpath"
            ),
            r"{\u0022token\u0022\u0009:\u0022TOPSECRET123\u0022}",
        )

        for encoded in diagnostics:
            for diagnostic in (encoded, quote(encoded, safe="")):
                with self.subTest(encoded=encoded, diagnostic=diagnostic):
                    redacted = safe_diagnostic_text(diagnostic)

                    self.assertIn("[redacted]", redacted)
                    self.assertNotIn("TOPSECRET123", redacted)

        deeply_encoded = r"Bearer\u0000TOPSECRET123"
        for _ in range(_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH + 2):
            deeply_encoded = json.dumps(deeply_encoded)[1:-1]

        depth_redacted = safe_diagnostic_text(deeply_encoded, max_chars=4_096)

        self.assertIn("[redacted", depth_redacted)
        self.assertNotIn("TOPSECRET123", depth_redacted)

    def test_safe_diagnostic_text_redacts_json_escaped_url_userinfo(self) -> None:
        cases = (
            (
                r'{"url":"https:\/\/operator:TOPSECRET123@provider.example\/path","status":"failed"}',
                "url",
            ),
            (
                r'{"detail":"https\u003a\/\/operator:TOPSECRET123@provider.example/path","status":"failed"}',
                "detail",
            ),
            (
                r'{"detail":"https://operator:TOPSECRET123\u0040provider.example/path","status":"failed"}',
                "detail",
            ),
            (
                r'{"detail":"\u0068ttps\u003A\/\/operator:TOP\u0053ECRET123'
                r'\u0040provider.example\/path","status":"failed"}',
                "detail",
            ),
        )

        for diagnostic, field in cases:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic)
                parsed = json.loads(redacted)

                self.assertEqual(parsed[field], "https://operator:[redacted]@provider.example/path")
                self.assertEqual(parsed["status"], "failed")
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_serialized_json_credential_fields(self) -> None:
        cases = (
            json.dumps({"token": "TOPSECRET123", "status": "denied"}, separators=(",", ":")),
            r'{"access\u005ftoken":"TOPSECRET123","status":"denied"}',
        )

        for inner in cases:
            with self.subTest(inner=inner):
                diagnostic = json.dumps(
                    {"detail": inner, "status": "failed"},
                    separators=(",", ":"),
                )

                redacted = safe_diagnostic_text(diagnostic)
                outer_result = json.loads(redacted)
                inner_result = json.loads(outer_result["detail"])

                self.assertEqual(inner_result["status"], "denied")
                credential_field = "access_token" if "access_token" in inner_result else "token"
                self.assertEqual(inner_result[credential_field], "[redacted credential]")
                self.assertEqual(outer_result["status"], "failed")
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_serialized_json_escaped_url_userinfo(self) -> None:
        inner = (
            r'{"url":"\u0068ttps\u003A\/\/operator:TOPSECRET123'
            r'\u0040provider.example\/path","status":"failed"}'
        )
        diagnostic = json.dumps({"detail": inner}, separators=(",", ":"))

        redacted = safe_diagnostic_text(diagnostic)
        inner_result = json.loads(json.loads(redacted)["detail"])

        self.assertEqual(inner_result["url"], "https://operator:[redacted]@provider.example/path")
        self.assertEqual(inner_result["status"], "failed")
        self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_escaped_json_fragments_inside_prose(self) -> None:
        cases = (
            (
                r"provider wrapper: {\"access\\u005ftoken\":\"TOPSECRET123\","
                r"\"status\":\"denied\"} end",
                "[redacted credential]",
            ),
            (
                r"provider wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:"
                r"TOPSECRET123\\u0040provider.example\\/path\",\"status\":\"denied\"} end",
                "operator:[redacted]",
            ),
        )

        for detail, expected in cases:
            with self.subTest(detail=detail):
                diagnostic = json.dumps(
                    {"detail": detail, "status": "failed"},
                    separators=(",", ":"),
                )

                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)
                outer_result = json.loads(redacted)

                self.assertTrue(outer_result["detail"].startswith("provider wrapper: "))
                self.assertTrue(outer_result["detail"].endswith(" end"))
                self.assertIn(expected, outer_result["detail"])
                self.assertEqual(outer_result["status"], "failed")
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_escaped_json_fragments_after_literal_quotes(self) -> None:
        details = (
            r'provider said "denied", wrapper: {\"access\\u005ftoken\":\"TOPSECRET123\"} end',
            (
                r'provider said "denied", wrapper: '
                r"\u007b\"access\\u005ftoken\":\"TOPSECRET123\"\u007d end"
            ),
            (
                r'provider said "denied", wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                r"TOPSECRET123\\u0040provider.example\\/path\"} end"
            ),
        )

        for detail in details:
            with self.subTest(detail=detail):
                diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)
                outer_result = json.loads(redacted)

                self.assertTrue(outer_result["detail"].startswith('provider said "denied", wrapper: '))
                self.assertTrue(outer_result["detail"].endswith(" end"))
                self.assertIn("[redacted", outer_result["detail"])
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_escaped_json_fragments_in_top_level_prose(self) -> None:
        diagnostics = (
            r"provider {\"access\\u005ftoken\":\"TOPSECRET123\"}",
            r'provider "denied" {\"access\\u005ftoken\":\"TOPSECRET123\"}',
            r'provider "denied" {\"access\\u005ftoken\":\"TOPSECRET123\"',
            (
                r'provider "denied" {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                r"TOPSECRET123\\u0040provider.example\\/path\"}"
            ),
        )

        for diagnostic in diagnostics:
            with self.subTest(diagnostic=diagnostic):
                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)

                self.assertIn("[redacted", redacted)
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_malformed_escaped_json_assignments(self) -> None:
        details = (
            r'provider said "denied", wrapper: {\"access\\u005ftoken\":\"TOPSECRET123\" end',
            (
                r'provider said "denied", wrapper: {"status":"failed",'
                r"\"access\\u005ftoken\":\"TOPSECRET123\"} end"
            ),
        )

        for detail in details:
            with self.subTest(detail=detail):
                diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)

                self.assertIsInstance(json.loads(redacted), dict)
                self.assertIn("[redacted credential]", redacted)
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_fails_closed_for_truncated_escaped_json_url(self) -> None:
        details = (
            (
                r'provider said "denied", wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                "TOPSECRET123"
            ),
            (
                r'provider said "denied", wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                r"TOPSECRET123\\u00"
            ),
            (
                r'provider said "denied", wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                r"TOPSECRET123\\"
            ),
            (
                r'provider said "denied", wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
                r"TOPSECRET123 C:\tail"
            ),
        )

        for detail in details:
            with self.subTest(detail=detail):
                diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))
                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)

                self.assertEqual(json.loads(redacted)["detail"], "[redacted credential]")
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_fails_closed_when_raw_quote_ends_escaped_url(self) -> None:
        diagnostics = (
            r'{\"url\":\"\\u0068ttps\\u003A\\/\\/operator:TOPSECRET123 "tail"',
            '{\\"url\\":\\"\\\\u0068ttps\\\\u003A\\\\/\\\\/operator:TOPSECRET123\n"tail"',
        )

        for original in diagnostics:
            diagnostic = original
            for depth in range(4):
                with self.subTest(original=original, depth=depth):
                    redacted = safe_diagnostic_text(diagnostic, max_chars=16_384)

                    self.assertIn("[redacted credential]", redacted)
                    self.assertNotIn("TOPSECRET123", redacted)
                    if depth:
                        self.assertIsInstance(json.loads(redacted), dict)
                diagnostic = json.dumps({"detail": diagnostic}, separators=(",", ":"))

    def test_safe_diagnostic_text_reconsiders_escaped_keys_after_unmatched_prefix_quote(self) -> None:
        diagnostics = (
            r"prefix \"unterminated before {\"access\\u005ftoken\":\"TOPSECRET123\"}",
            r"prefix \"access_token\":\"junk {\"access_token\":\"TOPSECRET123\"}",
            r'prefix \"unterminated "raw" before {\"access\\u005ftoken\":\"TOPSECRET123\"}',
            (
                r"prefix \"status\":\"unterminated before {"
                r"\"access\\u005ftoken\":\"TOPSECRET123\"}"
            ),
            (
                r'prefix \"unterminated before {"status":"failed",'
                r"\"access\\u005ftoken\":\"TOPSECRET123\"}"
            ),
            (
                r"prefix \"unterminated before {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:"
                r"TOPSECRET123\\u0040provider.example\\/path\"}"
            ),
        )

        for original in diagnostics:
            diagnostic = original
            for depth in range(3):
                with self.subTest(original=original, depth=depth):
                    redacted = safe_diagnostic_text(diagnostic, max_chars=16_384)

                    self.assertIn("[redacted", redacted)
                    self.assertNotIn("TOPSECRET123", redacted)
                    if depth:
                        self.assertIsInstance(json.loads(redacted), dict)
                diagnostic = json.dumps({"detail": diagnostic}, separators=(",", ":"))

    def test_safe_diagnostic_text_preserves_benign_malformed_escaped_json_lookalikes(self) -> None:
        details = (
            r'provider said "ok", value \"tokenized\":\"ordinary\" end',
            r'provider said "ok", value \"url_count\":\"2 end',
            (
                r'{"status":"ok"} \u0022status\u0022\u003a\u0022'
                r'token-bucket-filter-v2beta3\u0022 "tail"'
            ),
        )

        for detail in details:
            with self.subTest(detail=detail):
                diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

                self.assertEqual(safe_diagnostic_text(diagnostic, max_chars=4_096), diagnostic)

    def test_escaped_json_assignment_scan_does_not_rescan_incomplete_objects(self) -> None:
        repetitions = 400
        diagnostic = r"{\"a\":" * repetitions

        with patch(
            "leveraged_trader.output._decode_diagnostic_json_escape_unit",
            wraps=_decode_diagnostic_json_escape_unit,
        ) as decode_unit:
            redacted = _redact_escaped_json_assignments(
                diagnostic,
                secrets=set(),
                json_sensitive_values=set(),
                max_chars=len(diagnostic),
                depth=0,
            )

        self.assertEqual(redacted, diagnostic)
        self.assertLessEqual(decode_unit.call_count, repetitions * 4)

    def test_safe_diagnostic_text_redacts_multiply_escaped_json_fragment_inside_prose(self) -> None:
        detail = r"provider wrapper: {\"access\\u005ftoken\":\"TOPSECRET123\"} end"
        escape_layers = 3
        for _ in range(escape_layers):
            detail = json.dumps(detail, ensure_ascii=True)[1:-1]
        diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

        redacted = safe_diagnostic_text(diagnostic, max_chars=16_384)
        decoded_detail = json.loads(redacted)["detail"]
        for _ in range(escape_layers + 1):
            decoded_detail = json.loads(f'"{decoded_detail}"')

        self.assertIn('"access\\u005ftoken":"[redacted credential]"', decoded_detail)
        self.assertTrue(decoded_detail.startswith("provider wrapper: "))
        self.assertTrue(decoded_detail.endswith(" end"))
        self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_fails_closed_for_nested_json_prose_beyond_depth(self) -> None:
        diagnostic = r'provider wrapper: {"detail":"{\"access\\u005ftoken\":\"TOPSECRET123\"}"} end'
        for _ in range(_DIAGNOSTIC_NESTED_JSON_MAX_DEPTH + 1):
            diagnostic = json.dumps({"detail": diagnostic}, separators=(",", ":"))

        redacted = safe_diagnostic_text(diagnostic, max_chars=16_384)

        self.assertIsInstance(json.loads(redacted), dict)
        self.assertIn("[redacted credential]", redacted)
        self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_preserves_parent_json_across_escaped_url_depth(self) -> None:
        diagnostic = (
            r'provider "benign" wrapper: {\"url\":\"\\u0068ttps\\u003A\\/\\/operator:'
            r"TOPSECRET123\\u0040provider.example\\/path\"}"
        )

        for depth in range(1, _DIAGNOSTIC_NESTED_JSON_MAX_DEPTH + 3):
            diagnostic = json.dumps({"detail": diagnostic}, separators=(",", ":"))
            with self.subTest(depth=depth):
                redacted = safe_diagnostic_text(diagnostic, max_chars=65_536)

                self.assertIsInstance(json.loads(redacted), dict)
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_preserves_benign_escaped_json_fragment_inside_prose(self) -> None:
        detail = (
            r'provider said "ok", wrapper: {\"tokenized\":\"ordinary\",\"url\":'
            r"\"https:\\/\\/provider.example\\/path\"} end"
        )
        diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

        self.assertEqual(safe_diagnostic_text(diagnostic, max_chars=4_096), diagnostic)

    def test_safe_diagnostic_text_bounds_truncated_escaped_json_fragment_inside_prose(self) -> None:
        detail = r"provider wrapper: {\"access\\u005ftoken\":\"TOPSECRET123\"}" + "A" * 5_000
        diagnostic = json.dumps({"detail": detail}, separators=(",", ":"))

        redacted = safe_diagnostic_text(diagnostic, max_chars=64)

        self.assertEqual(redacted, "[redacted credential]")
        self.assertLessEqual(len(redacted), 64)
        self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_serialized_json_plain_assignments(self) -> None:
        for detail in (
            "Authorization: Bearer TOPSECRET123",
            "api_key=TOPSECRET123",
        ):
            with self.subTest(detail=detail):
                inner = json.dumps(
                    {"detail": detail, "status": "failed"},
                    separators=(",", ":"),
                )
                diagnostic = json.dumps(
                    {"detail": inner, "status": "outer"},
                    separators=(",", ":"),
                )

                redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)
                outer_result = json.loads(redacted)
                inner_result = json.loads(outer_result["detail"])

                self.assertEqual(inner_result["detail"], "[redacted credential]")
                self.assertEqual(inner_result["status"], "failed")
                self.assertEqual(outer_result["status"], "outer")
                self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_redacts_nested_unicode_escaped_sensitive_value(self) -> None:
        secret = "TOPSECRET123"
        unicode_escaped_secret = "".join(rf"\u{ord(character):04x}" for character in secret)
        inner = rf'{{"detail":"{unicode_escaped_secret}","status":"failed"}}'
        diagnostic = json.dumps({"detail": inner}, separators=(",", ":"))

        redacted = safe_diagnostic_text(
            diagnostic,
            sensitive_values=(secret,),
        )
        inner_result = json.loads(json.loads(redacted)["detail"])

        self.assertEqual(inner_result, {"detail": "[redacted]", "status": "failed"})
        self.assertNotIn(secret, redacted)
        self.assertNotIn(unicode_escaped_secret, redacted)

    def test_safe_diagnostic_text_preserves_benign_serialized_json(self) -> None:
        inner = json.dumps(
            {
                "status": "ok",
                "tokenized": "ordinary",
                "url": "https://provider.example/path",
            },
            separators=(",", ":"),
        )
        diagnostic = json.dumps(
            {"detail": inner, "status": "failed"},
            separators=(",", ":"),
        )

        self.assertEqual(safe_diagnostic_text(diagnostic, max_chars=4_096), diagnostic)

    def test_safe_diagnostic_text_bounds_truncated_serialized_json(self) -> None:
        inner = json.dumps(
            {"token": "TOPSECRET123", "padding": "A" * 5_000},
            separators=(",", ":"),
        )
        diagnostic = json.dumps({"detail": inner}, separators=(",", ":"))

        redacted = safe_diagnostic_text(diagnostic, max_chars=64)

        self.assertEqual(redacted, "[redacted credential]")
        self.assertLessEqual(len(redacted), 64)
        self.assertNotIn("TOPSECRET123", redacted)

    def test_safe_diagnostic_text_fails_closed_beyond_nested_json_depth(self) -> None:
        def wrap_serialized_json(value: str) -> str:
            return json.dumps({"detail": value}, separators=(",", ":"))

        base = json.dumps({"token": "DEEPSECRET123"}, separators=(",", ":"))
        at_limit = base
        for _ in range(_DIAGNOSTIC_NESTED_JSON_MAX_DEPTH):
            at_limit = wrap_serialized_json(at_limit)

        redacted_at_limit = safe_diagnostic_text(at_limit, max_chars=4_096)
        at_limit_result = json.loads(redacted_at_limit)
        for _ in range(_DIAGNOSTIC_NESTED_JSON_MAX_DEPTH):
            at_limit_result = json.loads(at_limit_result["detail"])
        self.assertEqual(at_limit_result["token"], "[redacted credential]")

        beyond_limit = wrap_serialized_json(at_limit)
        redacted_beyond_limit = safe_diagnostic_text(beyond_limit, max_chars=4_096)
        beyond_limit_result = json.loads(redacted_beyond_limit)
        for _ in range(_DIAGNOSTIC_NESTED_JSON_MAX_DEPTH):
            beyond_limit_result = json.loads(beyond_limit_result["detail"])
        self.assertEqual(beyond_limit_result["detail"], "[redacted credential]")
        self.assertNotIn("DEEPSECRET123", redacted_beyond_limit)

    def test_safe_diagnostic_text_bounds_oversized_url_userinfo_without_leaking_a_prefix(self) -> None:
        diagnostic = "provider said https://operator:" + "A" * 5_000 + "@provider.example/path"

        redacted = safe_diagnostic_text(diagnostic, max_chars=64)

        self.assertEqual(redacted, "provider said https://[redacted]")
        self.assertNotIn("AAAA", redacted)
        self.assertLessEqual(len(redacted), 64)

    def test_safe_diagnostic_text_preserves_benign_hyphenated_credential_prose(self) -> None:
        diagnostic = (
            "token-bucket-filter credential-helper-failed api-key-configuration-missing "
            "token-bucket-filter-v2beta3 credential-helper-failed-python3 "
            "api-key-configuration-missing-http401 credential-helper-failed-status403"
        )

        self.assertEqual(safe_diagnostic_text(diagnostic), diagnostic)

    def test_safe_diagnostic_text_scans_connected_vendor_prefix_once(self) -> None:
        # Exercise one full format_message-sized scan budget. This input used to
        # restart both assignment regexes after every dot and perform quadratic
        # work; keeping it as an ordinary functional test avoids a fragile
        # wall-clock threshold while still making that regression impractical.
        diagnostic = "A." * 8_192

        redacted = safe_diagnostic_text(diagnostic, max_chars=4_096)

        self.assertEqual(len(redacted), 4_096)
        self.assertTrue(redacted.endswith("…"))
        self.assertTrue(redacted.startswith("A.A.A."))

        credential = "Vendor." * 100 + "OpenAI.API.Key=TOPSECRET123"
        credential_redacted = safe_diagnostic_text(credential, max_chars=4_096)
        self.assertEqual(credential_redacted, "[redacted credential]")
        self.assertNotIn("TOPSECRET123", credential_redacted)

    def test_safe_diagnostic_text_still_redacts_standalone_opaque_credential_tokens(self) -> None:
        diagnostic = (
            "github-token-abCDef123456 github-token-abcdef123456 secret-abcdefghijklmnop local-secret "
            "github-token-http401 secret-python3 token-v2beta3"
        )

        redacted = safe_diagnostic_text(diagnostic)

        self.assertEqual(
            redacted,
            "[redacted credential] [redacted credential] [redacted credential] [redacted credential] "
            "[redacted credential] [redacted credential] [redacted credential]",
        )

    def test_safe_diagnostic_text_classifies_tokens_after_unicode_quote_escape(self) -> None:
        diagnostic = (
            r"provider reflected \u0022github-token-abCDef123456\u0022 and "
            r"\u0022token-v2beta3\u0022"
        )

        redacted = safe_diagnostic_text(diagnostic)

        self.assertEqual(
            redacted,
            r"provider reflected \u0022[redacted credential]\u0022 and "
            r"\u0022[redacted credential]\u0022",
        )

    def test_safe_diagnostic_text_preserves_noncredential_quoted_fields(self) -> None:
        diagnostic = (
            '{"status":"ok","tokenized":"word","api_keyboard":"layout",'
            '"client_secretary":"person","passwordless":"enabled","access_token_count":2,'
            '"clientSecretCount":2,"apiTokenCount":2,"xApiKeyboard":"layout",'
            '"authorizationHeaderPresent":false}'
        )

        self.assertEqual(safe_diagnostic_text(diagnostic), diagnostic)

    def test_safe_diagnostic_text_preserves_pascal_case_noncredential_labels(self) -> None:
        diagnostics = (
            "MyVendorClientSecretCount=2",
            "GoogleCloudTokenBucket=full",
            "OpenAIAPIKeyboard=layout",
            "GitHubAuthorizationHeaderPresent=false",
            '{"MyVendorClientSecretCount":2,"GoogleCloudTokenBucket":"full"}',
        )

        for diagnostic in diagnostics:
            with self.subTest(diagnostic=diagnostic):
                self.assertEqual(safe_diagnostic_text(diagnostic), diagnostic)

    def test_dataframe_cells_neutralize_provider_control_characters(self) -> None:
        output_buffer = io.StringIO()
        console = Console(
            file=output_buffer,
            record=True,
            width=200,
            color_system=None,
            no_color=True,
        )
        reporter = WorkflowReporter(console=console)

        reporter.universe_assets(
            pd.DataFrame(
                [
                    {
                        "symbol": "CTRL",
                        "name": "Issuer\x1b[2J\x00Fund",
                        "rsi_symbol": "QQQ",
                    }
                ]
            )
        )

        rendered = output_buffer.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x00", rendered)
        self.assertIn("Issuer [2J Fund", rendered)

    def test_asset_progress_includes_workflow_label(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=True, color_system=None, no_color=True)

        with patch("leveraged_trader.output.Progress") as mock_progress_type:
            progress = mock_progress_type.return_value
            progress.add_task.return_value = 7
            reporter = WorkflowReporter(console=console)

            with reporter.asset_progress(1, workflow_label="Short") as asset_progress:
                asset_progress.finish_asset()

        progress.add_task.assert_called_once_with("Processing Short Assets", total=1, status="")
        progress.update.assert_called_once_with(7, advance=1)

    def test_step_progress_tracks_terminal_workflow_steps(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=True, color_system=None, no_color=True)

        with patch("leveraged_trader.output.Progress") as mock_progress_type:
            progress = mock_progress_type.return_value
            progress.add_task.return_value = 7
            reporter = WorkflowReporter(console=console)

            with reporter.step_progress("Preparing startup", total=2) as step_progress:
                step_progress.start_step("Reconciling Alpaca positions")
                step_progress.finish_step()
                step_progress.start_step("Loading workflow assets")
                step_progress.finish_step()

        progress.add_task.assert_called_once_with("Preparing workflow", total=2, status="")
        progress.update.assert_has_calls(
            [
                call(7, status="Reconciling Alpaca positions"),
                call(7, advance=1),
                call(7, status="Loading workflow assets"),
                call(7, advance=1),
            ]
        )

    def test_step_progress_keeps_non_terminal_status_message(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        with reporter.step_progress("Reconciling Alpaca positions and loading workflow assets", total=2):
            pass

        self.assertIn(
            "Reconciling Alpaca positions and loading workflow assets",
            console.export_text(styles=False),
        )

    def test_default_reporter_uses_wide_width_for_non_terminal_output(self) -> None:
        output_buffer = io.StringIO()
        created_consoles: list[Console] = []

        def console_factory(**kwargs: object) -> Console:
            console = Console(file=output_buffer, record=True, color_system=None, **kwargs)
            created_consoles.append(console)
            return console

        with patch("leveraged_trader.output.Console", side_effect=console_factory):
            reporter = WorkflowReporter(no_color=True)

        self.assertEqual(len(created_consoles), 2)
        self.assertFalse(created_consoles[0].is_terminal)
        self.assertEqual(reporter.console.width, DEFAULT_NON_TERMINAL_WIDTH)

    def test_default_reporter_keeps_terminal_width_auto_sized(self) -> None:
        terminal_console = Console(file=io.StringIO(), force_terminal=True, color_system=None, no_color=True)

        with patch("leveraged_trader.output.Console", return_value=terminal_console) as mock_console:
            reporter = WorkflowReporter(no_color=True)

        mock_console.assert_called_once_with(no_color=True)
        self.assertIs(reporter.console, terminal_console)

    def test_injected_console_width_is_respected(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        self.assertIs(reporter.console, console)
        self.assertEqual(reporter.console.width, 120)

    def test_run_header_renders_cron_friendly_run_context(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        started_at_utc = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)

        reporter.run_header(
            started_at_utc=started_at_utc,
            mode="update",
            db_path="state.sqlite",
            output_dir="outputs",
            workflow_concurrency=4,
        )
        output = console.export_text(styles=False)
        lines = [line for line in output.splitlines() if line]
        expected_started_local = started_at_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")

        self.assertIn(f"Workflow Run: {expected_started_local}", output)
        self.assertNotIn("Started local", output)
        self.assertNotIn("Started UTC", output)
        self.assertNotIn("2026-01-02T14:30:00+00:00", output)
        self.assertIn("state.sqlite", output)
        self.assertIn("outputs", output)
        self.assertIn("4", output)
        self.assertTrue(lines[0].startswith("\u256d\u2500 Workflow Run: "))
        self.assertIn("\u2502 Download workers  4", output)
        self.assertTrue(lines[-1].startswith("\u2570"))
        self.assertTrue(lines[-1].endswith("\u256f"))
        self.assertNotEqual(lines[-1], "\u2500" * 100)

    def test_run_header_neutralizes_control_characters_in_runtime_paths(self) -> None:
        output_buffer = io.StringIO()
        console = Console(file=output_buffer, record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.run_header(
            started_at_utc=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
            mode="update",
            db_path="state\x1b[2J\x00.sqlite",
            output_dir="outputs\nFORGED HEADER",
            workflow_concurrency=4,
        )

        rendered = output_buffer.getvalue()
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x00", rendered)
        self.assertNotIn("outputs\nFORGED HEADER", rendered)
        self.assertIn("state [2J .sqlite", rendered)
        self.assertIn("outputs FORGED HEADER", rendered)

    def test_dataframe_caps_terminal_rows_with_caption(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.dataframe(
            "Limited Table",
            pd.DataFrame({"Asset": ["AAA", "BBB", "CCC"]}),
            [TableColumn("Asset", no_wrap=True)],
            empty_message="empty",
            max_rows=2,
            truncated_detail="full data written to limited.csv",
        )
        output = console.export_text(styles=False)

        self.assertIn("AAA", output)
        self.assertIn("BBB", output)
        self.assertNotIn("CCC", output)
        self.assertIn("Showing 2 of 3 rows; full data written to limited.csv.", output)

    def test_default_non_terminal_width_keeps_table_output_readable(self) -> None:
        output_buffer = io.StringIO()

        def console_factory(**kwargs: object) -> Console:
            return Console(file=output_buffer, record=True, color_system=None, **kwargs)

        with patch("leveraged_trader.output.Console", side_effect=console_factory):
            reporter = WorkflowReporter(no_color=True)

        reporter.reconciliation(
            pd.DataFrame(
                [
                    {
                        "Position ID": 1,
                        "Asset": "TQQQ",
                        "Action": "sell",
                        "Status": "submitted",
                        "Qty": 2,
                        "Limit Price": 150.0,
                        "Message": "submitted one-time GTC limit sell at frozen target price",
                    }
                ]
            )
        )
        output = reporter.console.export_text(styles=False)

        self.assertIn("Submitted GTC limit sell at frozen target price", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), DEFAULT_NON_TERMINAL_WIDTH)

    def test_reconciliation_table_wraps_message_without_truncating(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=80, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        reconciliation = pd.DataFrame(
            [
                {
                    "Position ID": 99,
                    "Display ID": 1,
                    "Asset": "TQQQ",
                    "Action": "sell",
                    "Status": "submitted",
                    "Buy Client Order ID": "rsi-buy-TQQQ-20260102",
                    "Sell Client Order ID": "rsi-exit-TQQQ-1",
                    "Qty": 2,
                    "Limit Price": 150.0,
                    "Alpaca Order ID": "alpaca-sell-order-1",
                    "Message": "submitted one-time GTC limit sell at frozen target price",
                }
            ]
        )

        reporter.reconciliation(reconciliation)
        output = console.export_text(styles=False)

        self.assertIn("Submitted GTC limit sell", output)
        self.assertIn("frozen target price", output)
        self.assertNotIn("ta...", output)
        self.assertNotIn("rsi-buy-TQQQ", output)
        self.assertNotIn("rsi-exit-TQQQ", output)
        self.assertNotIn("alpaca-sell-order-1", output)
        self.assertNotIn("99", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 80)

    def test_asset_run_summary_sorts_by_asset_and_uses_custom_title(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.asset_run_summary(
            [
                {
                    "Workflow #": 1,
                    "Workflow": "Long",
                    "Asset": "BBB",
                    "RSI Symbol": "BBB",
                    "Action": "Updating",
                    "Rows": 20,
                    "Status": "done",
                    "Message": "Processed 20 rows",
                },
                {
                    "Workflow #": 2,
                    "Workflow": "Long",
                    "Asset": "AAA",
                    "RSI Symbol": "AAA",
                    "Action": "Updating",
                    "Rows": 10,
                    "Status": "done",
                    "Message": "Processed 10 rows",
                },
            ],
            title="Long Asset Run Summary",
        )
        output = console.export_text(styles=False)

        self.assertIn("Long Asset Run Summary", output)
        self.assertNotIn("Workflow", output)
        self.assertLess(output.index("AAA"), output.index("BBB"))

    def test_universe_assets_renders_as_rich_table(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "rsi_symbol": "QQQ",
                }
            ]
        )
        universe.attrs["universe_title"] = "All Long Leveraged ETFs From Nasdaq Universe"
        universe.attrs["universe_counts"] = {
            "Current ETFs in Nasdaq table": 1,
            "Merged current ETFs/ETNs": 1,
            "Current long leveraged ETFs/ETNs found": 1,
            "Executable long leveraged ETFs/ETNs selected": 1,
            "RSI mappings needing review": 0,
            "Audit rows parsed": 10,
        }
        universe.attrs["universe_db_path"] = "state.sqlite"

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("All Long Leveraged ETFs From Nasdaq Universe", output)
        self.assertIn("Current long leveraged ETFs/ETNs found", output)
        self.assertIn("Executable long leveraged ETFs/ETNs selected", output)
        self.assertIn("RSI mappings needing review", output)
        self.assertNotIn("Current ETFs in Nasdaq table", output)
        self.assertNotIn("Merged current ETFs/ETNs", output)
        self.assertNotIn("Audit rows parsed", output)
        self.assertNotIn("Saved SQLite tables", output)
        self.assertIn("Asset", output)
        self.assertIn("Name", output)
        self.assertIn("RSI", output)
        self.assertIn("TQQQ", output)
        self.assertIn("ProShares UltraPro QQQ", output)
        self.assertIn("QQQ", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_universe_assets_renders_all_rows(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": f"A{index:03d}",
                    "name": f"Leveraged ETF {index:03d}",
                    "rsi_symbol": f"R{index:03d}",
                }
                for index in range(76)
            ]
        )

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("A000", output)
        self.assertIn("A074", output)
        self.assertIn("A075", output)
        self.assertNotIn("Showing 75 of 76 rows", output)

    def test_universe_assets_sorts_all_workflows_by_asset_symbol(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {"workflow": "Long", "symbol": "XYZG", "name": "Long XYZ", "rsi_symbol": "XYZ"},
                {"workflow": "Short", "symbol": "AIQD", "name": "Short AIQ", "rsi_symbol": "AIQ"},
                {"workflow": "Long", "symbol": "AALG", "name": "Long AAL", "rsi_symbol": "AAL"},
                {"workflow": "Short", "symbol": "AMZO", "name": "Short AMZN", "rsi_symbol": "AMZN"},
            ]
        )

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        positions = [output.index(symbol) for symbol in ("AALG", "AIQD", "AMZO", "XYZG")]
        self.assertEqual(positions, sorted(positions))

    def test_universe_assets_renders_failed_source_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "rsi_symbol": "QQQ",
                }
            ]
        )
        universe.attrs["universe_degraded"] = True
        universe.attrs["workflow_source_failures"] = [
            {
                "source": "Direxion",
                "source_type": "issuer_etf",
                "status": "source_error",
                "error": "HTTPError: 403 Client Error: Forbidden",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Universe is degraded", output)
        self.assertIn("Failed Workflow Universe Sources", output)
        self.assertIn("Direxion", output)
        self.assertIn("source_error", output)
        self.assertIn("HTTPError: 403", output)

    def test_universe_assets_renders_active_listing_failure_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "rsi_symbol": "QQQ",
                }
            ]
        )
        universe.attrs["universe_degraded"] = True
        universe.attrs["active_listing_source_failures"] = [
            {
                "source": "other_listed",
                "status": "error",
                "error": "offline",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Universe is degraded", output)
        self.assertIn("Failed Active Listing Sources", output)
        self.assertIn("other_listed", output)
        self.assertIn("Offline", output)

    def test_universe_assets_renders_audit_source_failure_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "rsi_symbol": "QQQ"}])
        universe.attrs["universe_degraded"] = True
        universe.attrs["universe_counts"] = {"Audit sources failed": 1}
        universe.attrs["audit_source_failures"] = [
            {
                "source": "NYSE exchange-traded products directory",
                "source_type": "exchange_directory",
                "status": "error",
                "error": "Parser returned no rows from an enabled audit source.",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Audit sources failed", output)
        self.assertIn("Failed Audit Universe Sources", output)
        self.assertIn("NYSE exchange-traded products directory", output)
        self.assertIn("Parser returned no rows", output)

    def test_universe_assets_renders_rsi_mapping_review_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "FOOU",
                    "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                    "rsi_symbol": "FOOU",
                }
            ]
        )
        universe.attrs["rsi_mapping_review"] = [
            {
                "symbol": "FOOU",
                "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                "rsi_symbol": "FOOU",
                "mapping_reason": "single-stock-style product did not expose a reliable underlying ticker",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Some RSI mappings need review", output)
        self.assertIn("RSI Mappings Needing Review", output)
        self.assertIn("FOOU", output)
        self.assertIn("Single-stock-style product", output)

    def test_optimization_summary_excludes_rows_below_trade_and_sharpe_thresholds(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.optimization_summary(
            pd.DataFrame(
                [
                    {
                        "Asset": "STXX",
                        "RSI Symbol": "STX",
                        "Start Date": "2026-01-02",
                        "Trading Days": 44,
                        "Buy RSI": 20.0,
                        "Sell Return Multiple": 1.1,
                        "Trades Executed": 0,
                        "Total Return": 0.0,
                        "CAGR": 0.0,
                        "Sharpe": -1751.799,
                        "Kelly Fraction": 0.0,
                        "Max Drawdown": 0.0,
                    }
                ]
            )
        )
        output = console.export_text(styles=False)

        self.assertIn("No strategies with at least 2 trades and Sharpe >= 1.0.", output)
        self.assertNotIn("STXX", output)
        self.assertNotIn("N/A", output)
        self.assertNotIn("-1751", output)

    def test_optimization_summary_uses_side_title_without_workflow_column(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        row = {
            "Workflow": "Long",
            "Asset": "TQQQ",
            "RSI Symbol": "QQQ",
            "Trades Executed": 2,
            "Sharpe": 1.25,
        }

        reporter.optimization_summary(
            pd.DataFrame([row]),
            title="Best Sharpe Parameters By Asset — Long",
        )
        output = console.export_text(styles=False)

        self.assertIn("Best Sharpe Parameters By Asset — Long", output)
        self.assertIn("TQQQ", output)
        self.assertNotIn("Workflow", output)

    def test_optimization_summary_shows_only_rows_with_enough_trades_and_sharpe(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        rows = [
            {
                "Asset": "GOOD",
                "RSI Symbol": "GD",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 1.0,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "GREAT",
                "RSI Symbol": "GT",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.2,
                "CAGR": 0.3,
                "Sharpe": 1.25,
                "Kelly Fraction": 0.5,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "LOW",
                "RSI Symbol": "LW",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 0.9999,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "ONE",
                "RSI Symbol": "ON",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 1,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 5.0,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
        ]
        rows.extend(
            [
                {
                    "Asset": f"A{index:03d}",
                    "RSI Symbol": f"R{index:03d}",
                    "Start Date": "2026-01-02",
                    "Trading Days": 2,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Total Return": 0.1,
                    "CAGR": 0.2,
                    "Sharpe": 0.8,
                    "Kelly Fraction": 0.4,
                    "Max Drawdown": -0.1,
                }
                for index in range(76)
            ]
        )

        reporter.optimization_summary(pd.DataFrame(rows))
        output = console.export_text(styles=False)

        self.assertIn("GOOD", output)
        self.assertIn("GREAT", output)
        self.assertNotIn("LOW", output)
        self.assertNotIn("ONE", output)
        self.assertNotIn("A075", output)
        self.assertNotIn("Showing", output)

    def test_signal_report_keeps_data_sufficiency_columns_when_wide(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.signal_report(
            "Buy Signals For Next Open",
            pd.DataFrame(
                [
                    {
                        "Asset": "ONX",
                        "RSI Symbol": "ON",
                        "Date": "2026-06-26",
                        "RSI Observation Date": "2026-06-28",
                        "Start Date": "2026-05-28",
                        "Trading Days": 21,
                        "Latest RSI": 37.25,
                        "Buy RSI": 48.0,
                        "Sell Return Multiple": 1.2,
                        "Trades Executed": 2,
                        "Sharpe": 5.7836,
                        "In Position": False,
                        "Pending Action": "buy",
                    }
                ]
            ),
            empty_message="empty",
        )
        output = console.export_text(styles=False)

        self.assertIn("Start", output)
        self.assertIn("Latest RSI", output)
        self.assertIn("RSI Date", output)
        self.assertIn("2026-05-28", output)
        self.assertIn("2026-06-28", output)
        self.assertIn("37.25", output)

    def test_combined_signal_report_shows_workflow_column(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.signal_report(
            "Buy Signals For Next Open",
            pd.DataFrame(
                [
                    {
                        "Workflow": "Short",
                        "Asset": "SQQQ",
                        "RSI Symbol": "QQQ",
                        "Date": "2026-06-26",
                        "Start Date": "2026-05-28",
                        "Trading Days": 21,
                        "Latest RSI": 72.25,
                        "Buy RSI": 70.0,
                        "Sell Return Multiple": 1.2,
                        "Trades Executed": 2,
                        "Sharpe": 5.7836,
                        "In Position": False,
                        "Pending Action": "buy",
                    }
                ]
            ),
            empty_message="empty",
        )
        output = console.export_text(styles=False)

        self.assertIn("Workflow", output)
        self.assertIn("Short", output)
        self.assertIn("SQQQ", output)

    def test_signal_report_omits_some_columns_when_narrow(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.signal_report(
            "Buy Signals For Next Open",
            pd.DataFrame(
                [
                    {
                        "Asset": "ONX",
                        "RSI Symbol": "ON",
                        "Date": "2026-06-26",
                        "RSI Observation Date": "2026-06-28",
                        "Start Date": "2026-05-28",
                        "Trading Days": 21,
                        "Latest RSI": 37.25,
                        "Buy RSI": 48.0,
                        "Sell Return Multiple": 1.2,
                        "Trades Executed": 2,
                        "Sharpe": 5.7836,
                        "In Position": False,
                        "Pending Action": "buy",
                    }
                ]
            ),
            empty_message="empty",
        )
        output = console.export_text(styles=False)

        self.assertNotIn("Start", output)
        self.assertNotIn("Latest RSI", output)
        self.assertNotIn("2026-05-28", output)
        self.assertNotIn("37.25", output)
        self.assertIn("RSI Date", output)
        self.assertIn("2026-06-28", output)
        self.assertIn("Days", output)
        self.assertIn("21", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_buy_signal_eligibility_summary_counts_managed_and_live_skips(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC"]}),
            eligible_buy_signals=pd.DataFrame({"Asset": ["CCC"]}),
            order_results=pd.DataFrame(
                {
                    "Asset": ["AAA", "BBB", "CCC"],
                    "Status": ["managed", "held", "submitted"],
                }
            ),
        )
        output = console.export_text(styles=False)

        self.assertIn("Buy Signal Eligibility", output)
        self.assertIn("Eligible buy signals", output)
        self.assertIn("1 / 3", output)
        self.assertIn("Submitted/existing/filled buys", output)
        self.assertIn("Skipped: active managed", output)
        self.assertIn("Skipped: Alpaca/live preflight", output)
        self.assertNotIn("Eligible after active managed filter", output)
        self.assertNotIn("Submitted or existing Alpaca buys", output)
        self.assertNotIn("Skipped by Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_omits_zero_detail_rows(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame(),
            eligible_buy_signals=pd.DataFrame(),
            order_results=pd.DataFrame(),
        )
        output = console.export_text(styles=False)

        self.assertIn("Eligible buy signals", output)
        self.assertIn("0 / 0", output)
        self.assertNotIn("Submitted/existing/filled buys", output)
        self.assertNotIn("Skipped: active managed", output)
        self.assertNotIn("Skipped: Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_reports_only_new_managed_races_as_concurrent(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame(
                {
                    "Workflow": ["Long", "Long"],
                    "Asset": ["ALREADY", "RACED"],
                    "Date": ["2026-01-02", "2026-01-02"],
                }
            ),
            eligible_buy_signals=pd.DataFrame(
                {
                    "Workflow": ["Long"],
                    "Asset": ["RACED"],
                    "Date": ["2026-01-02"],
                }
            ),
            order_results=pd.DataFrame(
                {
                    "Workflow": ["Long", "Long"],
                    "Asset": ["ALREADY", "RACED"],
                    "Date": ["2026-01-02", "2026-01-02"],
                    "Status": ["managed", "managed"],
                }
            ),
        )
        output = console.export_text(styles=False)

        active_line = next(line for line in output.splitlines() if "Skipped: active managed" in line)
        concurrent_line = next(line for line in output.splitlines() if "Skipped: concurrently managed" in line)
        self.assertTrue(active_line.rstrip().endswith("1"), active_line)
        self.assertTrue(concurrent_line.rstrip().endswith("1"), concurrent_line)

    def test_buy_signal_eligibility_summary_separates_ambiguous_budget_and_failures(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC", "DDD"]}),
            eligible_buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC", "DDD"]}),
            order_results=pd.DataFrame(
                {
                    "Status": [
                        "submission_unknown",
                        "submission_pending",
                        "batch_budget_exhausted",
                        "identity_mismatch",
                    ]
                }
            ),
        )
        output = console.export_text(styles=False)

        self.assertIn("Pending/unknown broker outcome — do not retry", output)
        self.assertIn("Skipped: budget/size", output)
        self.assertIn("Failed: Alpaca/live", output)
        self.assertNotIn("Skipped: Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_counts_confirmed_fills_as_successful(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC", "DDD"]}),
            eligible_buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC", "DDD"]}),
            order_results=pd.DataFrame(
                {
                    "Status": [
                        "submitted",
                        "existing",
                        "filled",
                        "partially_filled",
                    ]
                }
            ),
        )
        output = console.export_text(styles=False)

        success_line = next(line for line in output.splitlines() if "Submitted/existing/filled buys" in line)
        self.assertTrue(success_line.rstrip().endswith("4"), success_line)
        self.assertNotIn("Skipped: Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_does_not_hide_risky_statuses_as_preflight_skips(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": list("ABCDEFG")}),
            eligible_buy_signals=pd.DataFrame({"Asset": list("ABCDEFG")}),
            order_results=pd.DataFrame(
                {
                    "Status": [
                        "pending_cancel",
                        "superseded",
                        "incomplete_fill_metadata",
                        "fill_quantity_regression",
                        "batch_aborted",
                        "done_for_day",
                        "new-status-from-broker",
                    ]
                }
            ),
        )
        output = console.export_text(styles=False)

        ambiguous_line = next(line for line in output.splitlines() if "Pending/unknown broker outcome" in line)
        review_line = next(line for line in output.splitlines() if "Requires broker review/reconciliation" in line)
        batch_abort_line = next(line for line in output.splitlines() if "Skipped: batch safety abort" in line)
        unclassified_line = next(line for line in output.splitlines() if "Unclassified broker outcome" in line)
        self.assertTrue(ambiguous_line.rstrip().endswith("3"), ambiguous_line)
        self.assertTrue(review_line.rstrip().endswith("2"), review_line)
        self.assertTrue(batch_abort_line.rstrip().endswith("1"), batch_abort_line)
        self.assertTrue(unclassified_line.rstrip().endswith("1"), unclassified_line)
        self.assertNotIn("Failed: Alpaca/live", output)
        self.assertNotIn("Skipped: Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_counts_only_explicit_benign_preflight_statuses(self) -> None:
        preflight_statuses = [
            "stale_observation",
            "duplicate_signal",
            "open_order",
            "open_sell_order",
            "held",
            "not_tradable",
            "inactive",
            "deferred",
        ]
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": list(range(len(preflight_statuses)))}),
            eligible_buy_signals=pd.DataFrame({"Asset": list(range(len(preflight_statuses)))}),
            order_results=pd.DataFrame({"Status": preflight_statuses}),
        )
        output = console.export_text(styles=False)

        preflight_line = next(line for line in output.splitlines() if "Skipped: Alpaca/live preflight" in line)
        self.assertTrue(preflight_line.rstrip().endswith("8"), preflight_line)
        self.assertNotIn("Unclassified broker outcome", output)

    def test_buy_signal_eligibility_summary_counts_unowned_existing_order_as_failure(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": ["TQQQ"]}),
            eligible_buy_signals=pd.DataFrame({"Asset": ["TQQQ"]}),
            order_results=pd.DataFrame({"Status": ["unowned_existing"]}),
        )
        output = console.export_text(styles=False)

        failure_line = next(line for line in output.splitlines() if "Failed: Alpaca/live" in line)
        self.assertTrue(failure_line.rstrip().endswith("1"), failure_line)
        self.assertNotIn("Unclassified broker outcome", output)
        self.assertEqual(STATUS_STYLES["unowned_existing"], "red")

    def test_realized_pnl_summary_renders_percentage(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.realized_pnl_summary(
            pd.DataFrame(
                [
                    {
                        "Workflow": "Long",
                        "Closed Positions": 2,
                        "Complete Closed Positions": 1,
                        "Incomplete Closed Positions": 1,
                        "Total Buy Cost": 200.0,
                        "Total Sell Value": 250.0,
                        "Realized P/L": 50.0,
                        "Realized P/L %": 25.0,
                    }
                ]
            )
        )
        output = console.export_text(styles=False)

        self.assertIn("Closed Managed Alpaca Realized P/L", output)
        self.assertIn("Workflow", output)
        self.assertIn("Long", output)
        self.assertIn("50.00", output)
        self.assertIn("25.00%", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_fatal_broker_statuses_render_as_errors(self) -> None:
        for status in (
            "identity_mismatch",
            "incomplete_order_metadata",
            "fill_quantity_regression",
            "position_quantity_mismatch",
            "managed_order_conflict",
        ):
            with self.subTest(status=status):
                self.assertEqual(STATUS_STYLES[status], "red")

    def test_closed_audit_correction_status_renders_as_success(self) -> None:
        self.assertEqual(STATUS_STYLES["corrected"], "green")
        self.assertEqual(STATUS_STYLES["broker_inactive"], "yellow")
        self.assertEqual(STATUS_STYLES["retained"], "cyan")

    def test_paused_nonterminal_broker_statuses_render_as_warnings(self) -> None:
        for status in ("done_for_day", "stopped", "suspended"):
            with self.subTest(status=status):
                self.assertEqual(STATUS_STYLES[status], "yellow")

    def test_settings_handles_empty_grid_values(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[],
            profit_target_values=[],
        )
        output = console.export_text(styles=False)

        self.assertNotIn("Mode", output)
        self.assertNotIn("SQLite database", output)
        self.assertNotIn("Workflow concurrency", output)
        self.assertIn("Buy RSI values", output)
        self.assertIn("Sell return multiples", output)
        self.assertIn("none", output)

    def test_settings_renders_long_and_short_rsi_grids(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[20.0, 21.0],
            short_buy_rsi_values=[70.0, 71.0],
            profit_target_values=[1.1],
        )
        output = console.export_text(styles=False)

        self.assertIn("Long buy RSI values", output)
        self.assertIn("Short buy RSI values", output)
        self.assertIn("20 to 21 step 1", output)
        self.assertIn("70 to 71 step 1", output)

    def test_settings_lists_irregular_grid_values_without_inventing_steps(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[20.0, 25.0, 40.0],
            profit_target_values=[1.1, 1.7, 3.0],
        )
        output = console.export_text(styles=False)

        self.assertIn("20, 25, 40", output)
        self.assertIn("1.1, 1.7, 3.0", output)
        self.assertNotIn("step 1", output)
        self.assertNotIn("step 0.1", output)

    def test_settings_preserves_fractional_regular_grid_values_and_steps(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[20.25, 20.75, 21.25],
            profit_target_values=[1.15, 1.25, 1.35],
        )
        output = console.export_text(styles=False)

        self.assertIn("20.25 to 21.25 step 0.5", output)
        self.assertIn("1.15 to 1.35 step 0.1", output)

    def test_settings_preserves_fractional_irregular_grid_values(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[20.25, 20.75, 22.0],
            profit_target_values=[1.15, 1.25, 1.4],
        )
        output = console.export_text(styles=False)

        self.assertIn("20.25, 20.75, 22", output)
        self.assertIn("1.15, 1.25, 1.4", output)

    def test_workflow_footer_renders_elapsed_time_and_divider(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.workflow_footer(65.25)
        output = console.export_text(styles=False)
        lines = output.splitlines()

        self.assertIn("Workflow finished in 1m 05.25s.", output)
        self.assertNotIn("Workflow Benchmark", output)
        self.assertNotIn("CPU", output)
        self.assertNotIn("Peak RSS", output)
        self.assertEqual(lines[-1], "\u2500" * 100)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_format_duration_renders_all_nonfinite_values_as_empty(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), "nan", "inf", "-Infinity"):
            with self.subTest(value=value):
                self.assertEqual(format_duration(value), "")

    def test_numeric_formatters_render_all_nonfinite_values_as_empty(self) -> None:
        for formatter in (format_int, format_decimal_2):
            for value in (float("nan"), float("inf"), float("-inf"), "nan", "inf", "-Infinity"):
                with self.subTest(formatter=formatter.__name__, value=value):
                    self.assertEqual(formatter(value), "")

    def test_nonzero_count_treats_nonfinite_values_as_empty(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf"), "nan", "inf", "-Infinity"):
            with self.subTest(value=value):
                self.assertFalse(_is_nonzero_count(value))


if __name__ == "__main__":
    unittest.main()
