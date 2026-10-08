ablation2archive
2026-03-19T01:52:39.554511
2026-03-22T12:21:47.162637
共用时82个小时

除run_exp之外共使用594万token,累计2.76h
run_exp累计8900万，累计58.75h

✓ Loaded 80325 lines from /Users/zelinwu/PycharmProjects/LLMforpolymer/AIscientist/archive/ablation2archive/full.log

====================================================================================================
AI SCIENTIST LOG ANALYSIS REPORT
====================================================================================================

📊 OVERALL STATISTICS
   Total Days: 7
   Total Tokens: 5,945,429
   Total LLM Time: 221425.2s (61.51h)

📈 BY AGENT TYPE (Aggregated across all runs)
----------------------------------------------------------------------------------------------------
Agent Type               Runs      Input   Output      Total     LLM(s)
----------------------------------------------------------------------------------------------------
associate_random            7    660,347   64,657    725,004     1973.2
associate_reason            7    591,525   52,255    643,780     1556.9
internalize                 8    564,939   39,971    604,910     1157.0
learn_read                 23  1,527,993   67,043  1,595,036     2084.3
learn_search                7    924,988   25,543    950,531      775.2
orchestrator                8  1,018,569   66,610  1,085,179     1904.8
run_experiment             28          0        0          0   211505.7
work                       10    326,630   14,359    340,989      468.2

🔧 TOOL CALLS BY AGENT TYPE
----------------------------------------------------------------------------------------------------

associate_random:
   - search_memory: 33
   - read_memory: 20
   - write_memory_with_day: 1

associate_reason:
   - search_memory: 27
   - read_memory: 26
   - write_memory_with_day: 4

internalize:
   - read_memory: 22
   - search_memory: 17
   - write_integration_with_day: 3
   - update_memory_content: 1

learn_read:
   - read_memory: 42

learn_search:
   - search_arxiv: 45
   - read_memory: 14
   - download_to_day_dir: 14
   - copy_pool_papers_to_day: 5

orchestrator:
   - read_memory: 27
   - run_experiment: 5
   - work: 3
   - learn_read: 1
   - write_memory_with_day: 1

work:
   - read_memory: 50
   - search_memory: 8
   - best_of_n_select: 2

📅 DAILY BREAKDOWN
----------------------------------------------------------------------------------------------------

DAY_1:
   Tokens: 648,137 | LLM Time: 4156.4s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 46,010 tokens, 203.4s
   BEHAVIOUR RUNS (10 runs):
      ├─ learn_search: 141,352 tokens, 111.5s
      ├─ learn_read: 64,842 tokens, 79.6s
      ├─ learn_read: 53,023 tokens, 80.2s
      ├─ learn_read: 79,788 tokens, 96.7s
      ├─ learn_read: 75,670 tokens, 84.2s
      ├─ work: 11,134 tokens, 29.6s
      ├─ run_experiment: 0 tokens, 2687.5s
      ├─ internalize: 47,320 tokens, 154.5s
      ├─ associate_reason: 61,017 tokens, 255.0s
      └─ associate_random: 67,981 tokens, 374.2s

DAY_2:
   Tokens: 627,422 | LLM Time: 8147.0s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 59,372 tokens, 180.3s
   BEHAVIOUR RUNS (10 runs):
      ├─ learn_search: 117,327 tokens, 137.1s
      ├─ learn_read: 52,136 tokens, 87.1s
      ├─ learn_read: 79,861 tokens, 96.9s
      ├─ learn_read: 69,972 tokens, 84.0s
      ├─ learn_read: 52,727 tokens, 105.1s
      ├─ work: 19,070 tokens, 51.1s
      ├─ run_experiment: 0 tokens, 6608.1s
      ├─ internalize: 50,732 tokens, 155.7s
      ├─ associate_reason: 48,902 tokens, 292.7s
      └─ associate_random: 77,323 tokens, 348.8s

DAY_3:
   Tokens: 943,525 | LLM Time: 13410.0s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 252,443 tokens, 315.4s
   BEHAVIOUR RUNS (18 runs):
      ├─ learn_search: 61,394 tokens, 85.5s
      ├─ learn_read: 77,637 tokens, 97.7s
      ├─ learn_read: 85,464 tokens, 112.5s
      ├─ learn_read: 65,625 tokens, 103.2s
      ├─ learn_read: 83,507 tokens, 109.5s
      ├─ work: 24,717 tokens, 46.9s
      ├─ run_experiment: 0 tokens, 3127.1s
      ├─ run_experiment: 0 tokens, 4361.8s
      ├─ run_experiment: 0 tokens, 1502.8s
      ├─ run_experiment: 0 tokens, 123.4s
      ├─ run_experiment: 0 tokens, 2589.3s
      ├─ run_experiment: 0 tokens, 26.8s
      ├─ run_experiment: 0 tokens, 81.9s
      ├─ run_experiment: 0 tokens, 134.7s
      ├─ run_experiment: 0 tokens, 39.9s
      ├─ internalize: 93,618 tokens, 185.8s
      ├─ associate_reason: 84,218 tokens, 144.3s
      └─ associate_random: 114,902 tokens, 221.6s

DAY_4:
   Tokens: 842,340 | LLM Time: 52955.0s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 65,249 tokens, 187.6s
   BEHAVIOUR RUNS (10 runs):
      ├─ learn_search: 122,542 tokens, 103.4s
      ├─ learn_read: 77,978 tokens, 98.0s
      ├─ learn_read: 57,045 tokens, 89.8s
      ├─ learn_read: 86,384 tokens, 74.7s
      ├─ learn_read: 80,785 tokens, 87.8s
      ├─ work: 31,522 tokens, 50.8s
      ├─ run_experiment: 0 tokens, 51703.7s
      ├─ internalize: 66,889 tokens, 169.8s
      ├─ associate_reason: 149,938 tokens, 195.8s
      └─ associate_random: 104,008 tokens, 193.6s

DAY_5:
   Tokens: 845,584 | LLM Time: 6069.1s
   ORCHESTRATOR RUNS (2 runs):
      ├─ run_1: 136,359 tokens, 287.3s
      ├─ run_2: 47,100 tokens, 76.7s
   BEHAVIOUR RUNS (13 runs):
      ├─ learn_search: 142,175 tokens, 133.9s
      ├─ learn_read: 36,018 tokens, 82.4s
      ├─ learn_read: 74,958 tokens, 96.8s
      ├─ work: 32,089 tokens, 42.3s
      ├─ run_experiment: 0 tokens, 1391.9s
      ├─ run_experiment: 0 tokens, 3020.8s
      ├─ work: 30,749 tokens, 53.9s
      ├─ run_experiment: 0 tokens, 0.0s
      ├─ work: 40,327 tokens, 46.2s
      ├─ internalize: 85,515 tokens, 145.5s
      ├─ internalize: 19,811 tokens, 75.8s
      ├─ associate_reason: 125,624 tokens, 278.8s
      └─ associate_random: 74,859 tokens, 336.7s

DAY_6:
   Tokens: 839,957 | LLM Time: 104694.8s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 290,823 tokens, 360.1s
   BEHAVIOUR RUNS (16 runs):
      ├─ work: 79,201 tokens, 61.4s
      ├─ learn_search: 82,881 tokens, 96.4s
      ├─ learn_read: 55,863 tokens, 76.7s
      ├─ learn_read: 77,189 tokens, 99.2s
      ├─ work: 40,104 tokens, 49.7s
      ├─ run_experiment: 0 tokens, 2523.3s
      ├─ run_experiment: 0 tokens, 9613.3s
      ├─ run_experiment: 0 tokens, 7447.9s
      ├─ run_experiment: 0 tokens, 1300.0s
      ├─ run_experiment: 0 tokens, 10084.4s
      ├─ run_experiment: 0 tokens, 72000.2s
      ├─ run_experiment: 0 tokens, 162.6s
      ├─ run_experiment: 0 tokens, 302.4s
      ├─ internalize: 84,838 tokens, 136.5s
      ├─ associate_reason: 80,383 tokens, 196.6s
      └─ associate_random: 48,675 tokens, 184.1s

DAY_7:
   Tokens: 1,198,464 | LLM Time: 31992.9s
   ORCHESTRATOR RUNS (1 runs):
      ├─ run_1: 187,823 tokens, 293.9s
   BEHAVIOUR RUNS (13 runs):
      ├─ learn_search: 282,860 tokens, 107.4s
      ├─ learn_read: 70,233 tokens, 82.3s
      ├─ learn_read: 44,927 tokens, 73.5s
      ├─ learn_read: 93,404 tokens, 86.5s
      ├─ work: 32,076 tokens, 36.2s
      ├─ run_experiment: 0 tokens, 18446.6s
      ├─ run_experiment: 0 tokens, 1381.4s
      ├─ run_experiment: 0 tokens, 1506.1s
      ├─ run_experiment: 0 tokens, 1417.0s
      ├─ run_experiment: 0 tokens, 7920.9s
      ├─ internalize: 156,187 tokens, 133.4s
      ├─ associate_reason: 93,698 tokens, 193.7s
      └─ associate_random: 237,256 tokens, 314.1s

====================================================================================================

✓ JSON summary saved to: /Users/zelinwu/PycharmProjects/LLMforpolymer/AIscientist/archive/ablation2archive/log_analysis.json