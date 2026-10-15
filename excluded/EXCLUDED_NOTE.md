# Excluded material (kept for transparency, not used for any reported number)
- The first workbook returned by annotator B for round 2 (file withheld: it contains names of unrelated institutions and questions that are not part of this study; SHA-256 30c3c737a2fc3dfe5f0a2da61720a1bdc0770f50de10cc7dd3649e0d2147325f). 207 of its 208 rows did not
  match the study's questions, so it was set aside; B's labels were instead taken from a table B pasted back, transcribed in
  `labelling_ext/completed/annotator_B_labels_from_chat_paste.tsv` and checked row by row against the study items. The paper discloses this.
  How the first file was produced was not established.
- `discarded_fp32_attempt/`, `discarded_fp32_partial/`: partial float32 outputs from runs that were stopped (memory swapping)
  and redone in bfloat16 (see `grounding/GROUNDING_PROTOCOL.md` amendments). They were discarded without being analyzed.
