# Interview guide: Hybrid Intent Router

Everything in the boxes marked RESULTS is generated from `results/` (`make tables`), and `make verify`
fails if it ever drifts. Numbers written in the answers below were copied from the same files.

## 30-second pitch

<!-- RESULTS:pitch:START -->
"I built an intent router for a voice bot. A small multilingual model (MuRIL), fine-tuned in plain PyTorch on English only, answers when it is confident; otherwise the message goes to Gemini. On 1,500 test messages in English, Hindi and Kannada the router scored 84.2% accuracy, against 84.1% for the LLM alone and 78.6% for the small model alone, while sending only 11.1% of messages to the LLM. That cuts the LLM cost per 1,000 messages from $0.2991 to $0.0331 and the average latency from 1344 ms to 186 ms. I served it with Flask, gunicorn and NGINX and load-tested it at 41.2 requests per second with 0 failures."
<!-- RESULTS:pitch:END -->

## 2-minute explanation

1. **Problem.** A voice bot must first find the intent of a message (60 intents such as `alarm_set`,
   `weather_query`). An LLM does this well, but every call is a network round trip and costs money.
2. **Small model.** I fine-tuned `google/muril-base-cased` (a BERT-style encoder pre-trained on 17 Indian
   languages and English) with my own linear head and my own PyTorch training loop, on **English only**.
3. **Zero-shot.** Because MuRIL learned shared meanings across languages during pre-training, the English
   classifier also works on Hindi and Kannada sentences it never saw. Lower than English, but far above chance.
4. **Calibration.** Raw softmax scores are over-confident, so I fitted one temperature T on English dev
   (temperature scaling). Now "90% confident" means right about 90% of the time.
5. **Routing.** If the calibrated confidence is at least t, the small model answers; otherwise Gemini answers.
   I chose t on the dev subset with a rule written down in advance: the fewest LLM calls while staying within
   1 percentage point of LLM-only accuracy. Then I measured once on the test subset.
6. **Serving.** Flask API, gunicorn with 3 worker processes (4 PyTorch threads each, 12 cores), NGINX in
   front on port 8080 as a normal user. Locust load test through NGINX. If the LLM fails, the app returns the
   small model's answer instead of an error.

Final numbers (test_sub, 500 sentences per language):

<!-- RESULTS:routing:START -->
| Language | Small only | LLM only | Router | Oracle | Sent to LLM | Router cost /1000 | LLM-only cost /1000 | Router avg latency | LLM-only latency |
|---|---|---|---|---|---|---|---|---|---|
| English | 88.2% | 86.2% | **90.6%** | 93.4% | 3.8% | $0.0114 | $0.2991 | 89 ms | 1344 ms |
| Hindi | 77.0% | 84.2% | **82.0%** | 90.8% | 13.0% | $0.0389 | $0.2991 | 212 ms | 1344 ms |
| Kannada | 70.6% | 81.8% | **80.0%** | 90.2% | 16.4% | $0.0491 | $0.2991 | 258 ms | 1344 ms |
| All 3 (pooled) | 78.6% | 84.1% | **84.2%** | 91.5% | 11.1% | $0.0331 | $0.2991 | 186 ms | 1344 ms |

test_sub, 500 messages per language. Threshold t = 0.56 was chosen on dev_sub (pooled dev accuracy 83.2%, target 83.2%, 14.3% sent to LLM). Temperature T = 1.342.
<!-- RESULTS:routing:END -->

## 20 likely questions, with short answers

1. **Why route at all, instead of always using the LLM?**
   Most messages are easy. The small model answers them on a CPU in about 38 ms (median 37.5 ms), for free. The
   LLM took a median of 1,344 ms per single call and is billed per token. Routing pays for the LLM only on the
   hard cases: 11.1% of test messages, with the same pooled accuracy as LLM-only (84.2% vs 84.1%).

2. **Why not use only the small model?**
   On Hindi and Kannada it is clearly weaker than the LLM (it never saw those languages in training):
   77.0% and 70.6% on test_sub, against 84.2% and 81.8% for the LLM. Pooled, small-only is 78.6% vs 84.1%.
   (On English it is actually better than the LLM: 88.2% vs 86.2%.)

3. **Why did you choose the threshold on dev data and not on test data?**
   If I tune anything on the test set, the test score is no longer an honest estimate of new data; it is
   optimistic. So every decision (prompt, temperature, threshold, epochs) used dev data, and the test subset
   was used once, at the end.

4. **What exactly is the threshold rule?**
   Try t = 0.00, 0.01, ..., 1.00 and 1.01. Keep the values whose pooled dev_sub accuracy is at least LLM-only
   accuracy minus 1 percentage point, then take the one with the fewest LLM calls. t = 1.01 sends everything to
   the LLM, so a valid choice always exists. LLM-only on pooled dev_sub was 84.2%, so the target was 83.2%,
   and the rule picked t = 0.56 (14.3% of dev messages to the LLM).

5. **Why does a model trained on English work for Kannada?**
   MuRIL was pre-trained on 17 Indian languages and English together, including translated and transliterated
   pairs. Sentences that mean the same thing end up with similar vectors in every language, so a classifier
   trained on English vectors partly works on Kannada vectors. This is called zero-shot cross-lingual transfer.

6. **Why MuRIL and not XLM-R?**
   Published zero-shot results on this exact dataset (IndicXTREME) are much better for MuRIL on Hindi and
   Kannada (82.4 / 77.5 vs 74.9 / 61.7). Both are 12-layer base-size encoders, so the speed is similar.

7. **Why freeze the word embeddings?**
   MuRIL's vocabulary has 197k tokens; that matrix is most of the parameters. Freezing it saves GPU memory on a
   4 GB card, and keeps the shared multilingual word vectors that make transfer work. If English fine-tuning
   moved only the English token vectors, Hindi and Kannada tokens would drift apart from them.

8. **Walk me through your training loop.**
   For each batch: tokenize (max 64 tokens), forward pass, cross-entropy loss, backward pass with a gradient
   scaler for fp16, clip gradients to norm 1.0, AdamW step, learning-rate scheduler step. After each epoch:
   English dev accuracy; keep the best checkpoint; stop after 2 epochs without improvement.

9. **What does temperature scaling do?**
   It divides all logits by one number T before the softmax. T > 1 makes the model less confident. It never
   changes which class has the highest score, so accuracy stays the same; only the confidence changes. I fitted
   T by minimising negative log-likelihood on English dev: T = 1.342 (the model was over-confident), and ECE on
   English dev fell from 0.076 to 0.016.

10. **What is ECE?**
    Expected calibration error: put predictions into confidence bins, and in each bin compare the average
    confidence with the actual accuracy. ECE is the weighted average gap. 0 means perfectly calibrated.

11. **Why did you fit the temperature on English, when the threshold is used for all languages?**
    To stay zero-shot: no Hindi or Kannada labels were used to build the small model. The threshold itself was
    then chosen on the pooled dev subset, which is allowed because it is dev data, not test data.

12. **How do you know the LLM numbers are reliable?**
    Every answer is cached with the model id and prompt version, the output is forced to one of the 60 labels
    (0 invalid answers in 2,400), and I measured that batching 20 messages per request did not change accuracy
    on 50 items (82.0% single vs 82.0% batched, 98% identical answers).

13. **How did you compute the cost?**
    From real single-call usage metadata: on average 890.3 input and 12.8 output tokens per call, times the paid
    price ($0.30 per million input, $2.50 per million output tokens, checked 2026-11-22), times the share of
    messages the router sends to the LLM.

14. **What happens if the LLM is down?**
    The API catches the error or timeout and returns the small model's answer with `routed_to = "small_fallback"`.
    The bot gets a slightly less accurate answer instead of an error. There is a test for it.

15. **Why gunicorn and NGINX? Why not just `flask run`?**
    Flask's built-in server is for development: one process, not hardened. gunicorn runs several worker
    processes, restarts crashed ones and enforces timeouts. NGINX in front handles slow clients, limits request
    size and keeps connections efficient.

16. **How did you choose 3 workers?**
    12 CPU cores and 7.4 GB RAM. Inference is CPU-bound, so 3 workers x 4 PyTorch threads = 12 threads, one per
    core, without the workers fighting over cores. `preload_app` loads the model once before forking.

17. **What is the bottleneck?**
    CPU-bound model inference. One message alone takes 37.5 ms, but under load (20 users, 41.2 requests/s,
    0 failures) each of the 3 workers spends about 73 ms per request because all 12 hardware threads are shared
    with the load generator, and requests queue (p50 180 ms, p95 260 ms). NGINX and Flask cost almost nothing.
    To go faster: INT8 quantization or more CPU, not more workers.

18. **What were the hardest bugs?**
    A network timeout type that my retry code did not catch, and a machine crash that left zero bytes in the
    answer cache. Both are fixed at the root and have regression tests (see "Errors" below).

19. **What are the limitations?**
    MASSIVE is voice-assistant data, not banking; the Hindi and Kannada sentences are clean translations, not
    real code-mixed speech; the LLM was tested on 800 sentences per language; everything ran on one laptop.

20. **What would you do next?**
    INT8 dynamic quantization for faster CPU inference; add a little Hindi and Kannada training data; test on
    real code-mixed speech transcribed with Whisper; learn the routing decision instead of a single threshold.

## Errors we hit and how we fixed them

1. **HTTP read timeout crashed the LLM run.** The SDK's HTTP library raised its own `ReadTimeout`, which my
   retry code did not catch. Fix: map every network error to a retried error; regression test added.
2. **The Linux VM stopped mid-run and corrupted the answer cache.** The file ended with 2,244 zero bytes and
   the next run crashed on load. Fix: `fsync` after every write; the loader drops crash padding but still fails
   on real corruption; regression test added. No paid answers were lost.
3. **Rate limits (429) and overload (503) from the API.** Handled by exponential backoff with jitter.
4. **The first training run stopped too early.** After 5 epochs English dev accuracy was still rising
   (77.4%). I raised the cap to 15 epochs and let early stopping on English dev decide. I kept run 1's log.
5. **gunicorn workers hung on every request.** `preload_app` loaded the model in the master, which started
   PyTorch's OpenMP thread pool; forked workers inherited a broken copy and deadlocked. Locust showed
   "0 requests", not errors. Fix: the master runs single-threaded (`torch.set_num_threads(1)`), workers set 4
   threads after the fork; plus a real `/predict` check in `make stack-up` and a 30 s timeout in the load test.
6. **`make` hid failures.** `cmd | tee log` returns tee's exit code; fixed with `pipefail`. Python buffered
   its output into the log; fixed with `PYTHONUNBUFFERED=1`.

## Resume bullets (numbers from results/)

<!-- RESULTS:resume:START -->
- **Software / ML resume:** Built a hybrid intent router in PyTorch: a MuRIL classifier fine-tuned on English only reached 77.9% (Hindi) and 70.7% (Kannada) zero-shot accuracy; temperature-scaled confidence routing sent only 11.1% of messages to an LLM (+0.1 pp accuracy vs LLM-only, 88.9% lower LLM cost); served with Flask, gunicorn and NGINX at 41.2 requests/s (p95 260 ms).
- **Data analyst resume:** Compared an LLM with a small model on 1,500 multilingual test messages; chose a routing threshold on a separate dev set from 102 candidates and showed the router keeps LLM-only accuracy (+0.1 percentage points) with 88.9% fewer LLM calls (LLM cost per 1,000 messages $0.2991 -> $0.0331); built the charts and auto-generated result tables that tests check against the raw results.
<!-- RESULTS:resume:END -->
