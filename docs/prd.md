# Product brief

Goal: preserve all inputs required to score all 40 released DWF submissions in one public GitHub repository, without modifying the live harness or exposing runtime credentials.

Success means the original scorer can dry-run all 40 exported run roots; each submission validates and has hash-bound registered inputs and frozen outputs; formal scoring can later run with an authorized judge endpoint. The original 27 exports and all original scoring assets remain byte-identical to the first published commit. The remaining 13 are an explicitly versioned, no-overwrite extension.

Privacy scope: reject credentials throughout the publishable tree and raw runtime logs in the new 13 exports. Preserve the two already-published Energy_03 Workflow journals and the original compiled scoring cache only as exact-hash legacy exceptions, per the owner's explicit instruction to leave the old package alone. Document those exceptions instead of claiming the entire repository has zero historical logs.

Out of scope: rescoring the scientific results here, changing benchmark rules, or exporting active/unreleased tasks.
