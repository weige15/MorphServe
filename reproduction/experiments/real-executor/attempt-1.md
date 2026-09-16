# Attempt 1 — free-block expectation bug

The real executor completed injected-failure rollback, activated `[25,24,26]`, selected explicit mapping, expanded three 615-block groups, recovered LIFO, restored exact FP16 logits, and preserved FCFS order. The command exited 1 only because the harness expected `num_free_blocks == num_blocks - 4`, incorrectly assuming the four original blocks were occupied. This protocol intentionally had no active requests, so all 1,849 blocks should be free.

The changed retry fixes only that assertion to require `num_free_blocks == num_blocks` and reruns the frozen actions.
