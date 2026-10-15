# Amendment 1 to EXTENSION_PREREGISTRATION.md (2026-10-01, before any generated answer was read)

Change: model weights are loaded in bfloat16 instead of float32.

Reason: purely technical. In float32 the 3B model (~13 GB) exceeded available RAM on the 17 GB machine; the process
swapped (0.1 GB RAM free, 10 GB swap used) and the first answer took 829 s, i.e. ~24 h per model. bfloat16 is the
precision in which both models' weights are released, halving memory (~6.4 GB for Llama).

Handling of the float32 output: exactly one answer (Llama, H01) had been generated in float32. Its content was not
read. It is moved, unread, to answers/discarded_fp32_attempt/ and not used. All 104 answers per model are generated
in bfloat16 with the same seeds and settings. Nothing else in the pre-registration changes.
