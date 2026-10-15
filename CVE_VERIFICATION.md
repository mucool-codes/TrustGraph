# CVE_VERIFICATION.md

Written in S13 (2026-10-15). Verification of the three CVE records listed in `TECHNICAL_AND_NOVELTY.md` (line 27 and the appendix, lines 247-249).
CLAIMS.md X13 bars these records until they are verified; this file is the evidence. **CLAIMS.md is not edited here** (S13 Task 3): the proposed wording change is in the S13 DECISION REQUEST.

## Method

Each record was read on **2026-10-15** from the CVE Program's record service, `https://cveawg.mitre.org/api/cve/<id>`, which serves the JSON (CVE record format 5.2) that cve.org displays for the same identifier (`https://www.cve.org/CVERecord?id=<id>`). The cve.org page itself is a script-rendered page and was not read; the record service is the same CVE Program data. NVD was not read. Nothing below is from memory: every "record says" is from the fetched JSON.

## Records

| CVE | URL read | State / dates (from the record) | What the record says (one line) |
|---|---|---|---|
| CVE-2026-37554 | https://cveawg.mitre.org/api/cve/CVE-2026-37554 (display page: https://www.cve.org/CVERecord?id=CVE-2026-37554) | PUBLISHED; reserved 2026-04-06; published 2026-05-01; updated 2026-05-07; assigner MITRE | In Vanetza V2X v26.02 remote unauthorized attackers can cause a denial of service: OpenSSL ECC point-validation exceptions escape the `Router::indicate()` call chain in the GeoNetworking packet pipeline, causing `std::terminate` and crashing the V2X receiver. CVSS 3.1 base 7.5 HIGH (`AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H`). CWE-248 appears in the CISA-ADP container, not in the CNA container (which says "n/a"). Affected vendor/product/version fields read "n/a". |
| CVE-2026-43988 | https://cveawg.mitre.org/api/cve/CVE-2026-43988 | PUBLISHED; reserved 2026-05-04; published 2026-05-26; updated 2026-05-27; assigner GitHub_M | "Vanetza: Remote Denial of Service via Uncaught Exception in ASN.1/OER Parsing": in 26.02 and earlier, malformed ASN.1/OER structures make `asn1c_wrapper.cpp` raise a `std::runtime_error` that is not caught at the parsing boundary and terminates the process. Fixed by Vanetza commit `62dfe58a8342512b6e1947d75821402ada524f1a`. CWE-248; CVSS 3.1 7.5 HIGH, same vector. Advisory GHSA-j6cj-rp87-mfrx. |
| CVE-2026-44905 | https://cveawg.mitre.org/api/cve/CVE-2026-44905 | PUBLISHED; reserved 2026-05-07; published 2026-05-26; updated 2026-05-27; assigner GitHub_M | "Vanetza: Remote Denial of Service via Uncaught OER Encoding Exception in Cryptographic Verification": in 26.02 and earlier, a certificate whose Psid sub-type violates its constraints is accepted at parsing, and the `std::runtime_error` raised later when `StraightVerifyService` re-encodes the signing certificate to hash it is uncaught, terminating the process. Fixed by commit `e1a2e2709210d309458c3d77f98d50dec26c0df0`. CWE-248; CVSS 3.1 7.5 HIGH, same vector. Advisory GHSA-q9fq-3rx9-7xcv. |

## What TECHNICAL_AND_NOVELTY.md says, checked against the records

| Statement in TECHNICAL_AND_NOVELTY.md | Status |
|---|---|
| CVE-2026-37554: Vanetza V2X v26.02, DoS via uncaught OpenSSL ECC point-validation exception escaping `Router::indicate()` to `std::terminate`; reserved 2026-04-06, published 2026-05-01, MITRE-assigned, CVSS 3.1 base 7.5 with the stated vector | **verified** |
| CVE-2026-43988: Vanetza <= 26.02, uncaught `std::runtime_error` in the ASN.1/OER parsing pipeline (CWE-248), fixed in commit `62dfe58` | **verified** (CWE-248 is in the CVE record's CNA container) |
| CVE-2026-44905: Vanetza <= 26.02, DoS in the cryptographic verification pipeline via unenforced Psid semantic constraints, fixed in commit `e1a2e27` | **verified** (the record says the constraints are not enforced at parsing and the failure occurs on re-encoding during verification) |
| The three are CWE-248 uncaught-exception DoS in the same stack, published within a month of each other | **verified** (all three Vanetza <= 26.02; published 2026-05-01, 2026-05-26, 2026-05-26; CWE-248 for the first comes from CISA-ADP) |
| "a single malformed packet crashing an **ETSI ITS-G5** receiver" (line 27) | **unverified as worded**: the record says a "V2X receiver" / GeoNetworking pipeline of the Vanetza implementation; the string "ITS-G5" does not appear in it |
| "with no credential compromise required" | **consistent with** "remote unauthorized attackers", PR:N; the wording is the document's, not the record's |
| "A node holding a valid certificate can be taken offline by any unauthenticated neighbour **within radio range**" (line 27) | **unverified**: the records give attack vector NETWORK and do not mention radio range, valid certificates or roadside deployments |
| That RSUs or deployed vehicular products run the affected code | **unverified**: the records concern the open-source Vanetza implementation (CNA fields for 37554 read "n/a"); no record names a deployed product |
| The record that the gist from Innora Security Research exists | not verified; the 37554 record lists the gist URL as a reference, which is not a verification of the gist's content |

## What may be said, if CLAIMS.md is amended (proposal, not applied)

Allowed with these citations: "CVE-2026-37554, CVE-2026-43988 and CVE-2026-44905 are published CVE records of denial-of-service vulnerabilities (CVSS 3.1 base 7.5) in the open-source Vanetza V2X implementation (<= 26.02), in each of which a remote unauthenticated network attacker can terminate the process through an uncaught exception (CWE-248); the first concerns ECC point validation in GeoNetworking packet processing, the second ASN.1/OER parsing, the third the cryptographic verification path." Not allowed from these records: ITS-G5, radio range, deployed RSU products, that such an attack has occurred in the field, that it affects certificate-holding nodes in particular, or any statement about SCMS revocation delay (X13 stays in force for everything not listed above).
