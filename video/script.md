# Tutorial video script: Hybrid Intent Router

How this file is used: `video/build_video.py` turns every "## " section into one scene (slide + narration).
Lines starting with "- " are slide bullets, "visual:" picks the image, and lines starting with "> " are
the narration. Numbers are written as {{file.json:key|format}} and filled from `results/`, so the video never
contains a number that the code did not produce. The narration is written normally; `build_video.py`
swaps words like PyTorch, NGINX and LLM for their spoken form ("Pie Torch", "engine X", "L L M").

## Hybrid Intent Router
- A small model answers first; a big LLM only when the small model is unsure
- Trained on English only, tested on Hindi and Kannada
- Plain PyTorch, Flask, gunicorn and NGINX on Linux
visual: image results/charts/architecture.png
> Hi. In this video I will show you, step by step, how I built the Hybrid Intent Router.
> It is a small multilingual model that understands what a customer wants, in English, Hindi and Kannada.
> When it is confident, it answers in milliseconds on a CPU. When it is not, it asks a large language model.
> I will show the real code, the real commands, and the real results from this run.

## The problem
- A voice bot must first find the customer's intent, for example alarm_set or weather_query
- Sending every message to an LLM is slow and costs money on every call
- A small model is fast and free to run, but makes more mistakes
- Indian users write in many languages, but labelled data is mostly English
visual: none
> A voice bot first has to understand what the customer wants. We call that the intent.
> For example, "wake me up at five am" is the intent alarm set.
> The easy answer is to send every message to a large language model. But every call is a network round trip, and every call costs money.
> A small model is fast and free to run, but it makes more mistakes. And most labelled training data is in English, while many users write in Hindi or Kannada.

## The idea: small model first, LLM when unsure
- Small model gives an intent and a confidence
- Confidence >= threshold t: keep the small model's answer
- Confidence < t: send the message to the LLM
- The threshold is chosen on dev data, never on test data
visual: image results/charts/architecture.png
> So here is the idea. Every message goes to the small model first. It returns an intent and a confidence between zero and one.
> If the confidence is above a threshold, which I call t, we keep the small model's answer. If it is below t, we send the message to the LLM.
> The whole project is about measuring this honestly. How many messages can the small model handle alone? What happens to accuracy? And how much time and money does it save?

## Phase 0: check the machine first
- WSL2 Ubuntu on Windows, 12 CPU cores, 7.4 GB RAM
- NVIDIA RTX 3050 laptop GPU with 4 GB for training
- The .gitignore with .env was committed first, so the API key can never reach git
visual: output logs/00_env.log 26-46
> Before writing any code, I checked the machine and saved the output to a log file.
> The project runs in WSL2 Ubuntu, because gunicorn does not run on native Windows.
> There are 12 CPU cores and 7.4 gigabytes of RAM, which later decide how many server workers I can run. The laptop also has a small NVIDIA GPU with 4 gigabytes, which is enough for training.
> The very first commit was the gitignore file, so the API key in the dot env file can never be committed by accident.

## Phase 1: research before code
- Dataset: Amazon MASSIVE 1.1, 60 intents, parallel sentences
- Small model: MuRIL base, built for Indian languages
- LLM: gemini-3.5-flash-lite with JSON output
- Routing idea: temperature scaling + a confidence threshold
visual: none
> Next, research. I looked up every fact I needed and saved each one with its source link and the date.
> The dataset is Amazon MASSIVE. The model is MuRIL, which is a BERT style model trained on seventeen Indian languages and English. Published results show it transfers to Hindi and Kannada better than XLM-R, the more popular choice.
> For the LLM, I picked Gemini 3.5 Flash Lite, because it is cheap, fast, and supports structured JSON output.

## Phase 2: the data
- MASSIVE: the same sentences translated into many languages
- English, Hindi, Kannada: 11,514 train, 2,033 dev, 2,974 test each
- The same id means the same sentence in every language
- SHA-256 check, so the data is exactly reproducible
visual: output logs/02_data.log 1-14
> MASSIVE is a voice assistant dataset from Amazon. Each sentence was translated into fifty-one languages, and the translations share the same id.
> That is very useful. It means I can test Hindi and Kannada on exactly the same sentences as English.
> The make data command downloads the official file, checks its SHA-256 hash, and writes clean files with the text, the intent and a number label.

## Phase 2: fixed subsets for decisions and for final numbers
- dev_sub: 300 dev sentences, used for every decision
- test_sub: 500 test sentences, used only for final numbers
- Same ids in all three languages, stratified by intent, seed 42
- Tests prove they never overlap with each other or with train
visual: code src/router/data.py 70-89
> The free tier of the LLM is limited, so the LLM cannot see all five thousand test sentences in three languages.
> I made two fixed subsets with a seed. Dev sub has 300 sentences and is used for every decision. Test sub has 500 sentences and is used only for the final numbers.
> The sampling is stratified, which means each intent gets a share proportional to its size. Nineteen unit tests check sizes, labels, parallel ids, and that the subsets never overlap.

## Phase 3: the LLM client
- Official google-genai SDK, generate_content
- JSON output whose intent must be one of the 60 labels
- Every answer is still validated; a bad one is retried once, then marked invalid
- Exponential backoff for rate limits and server errors
visual: code src/router/llm.py 68-96
> Phase three is the baseline: send everything to the LLM.
> I used the official google-genai SDK. The response must be JSON, and the intent field is an enum, which means a fixed list of the sixty labels.
> I still validate every answer. If a label is wrong, I retry once, and then count it as invalid, which means wrong.
> Rate limits and server errors are retried with exponential backoff, so the wait doubles each time.

## Phase 3: a cache that makes runs resumable
- Every answer is saved by model, prompt version, mode, language and id
- A key is never sent to the API twice
- Runs continue where they stopped, and analysis works without an API key
visual: code src/router/llm.py 269-302
> Every LLM answer is written to a cache file. The key is the model, the prompt version, the mode, the language and the sentence id.
> So if a run stops half way, I just run it again, and only the missing sentences are sent.
> The cache is also committed to git, so anyone can reproduce the routing analysis without an API key.

## Phase 3: picking the prompt on dev data
- Version 1: the list of labels only
- Version 2: labels plus one short English example per intent
- Pilot: 60 dev_sub sentences, 20 per language
- Version 1: {{llm_pilot.json:versions.v1_labels.pooled|pct}}, version 2: {{llm_pilot.json:versions.v2_labels_examples.pooled|pct}}; version 2 was frozen
visual: output logs/03_llm_pilot.log
> I tried two prompts on sixty dev sentences. The first gives only the list of labels. The second also gives one short English example for each intent, taken from the training data.
> The version with examples scored {{llm_pilot.json:versions.v2_labels_examples.pooled|pct}}, against {{llm_pilot.json:versions.v1_labels.pooled|pct}} without them. So version two was frozen for all later runs.
> This decision used dev data only. The test data stays untouched until the end.

## Phase 3: the LLM baseline
- 2,400 classifications: 300 + 500 sentences x 3 languages
- 20 messages per request to stay inside the free tier
- dev_sub, all 3 languages: {{llm_baseline.json:accuracy.pooled_dev_sub|pct}}
- test_sub, all 3 languages: {{llm_baseline.json:accuracy.pooled_test_sub|pct}}
- Invalid answers: {{llm_baseline.json:invalid.hi_test_sub|int}}
visual: output logs/03_llm_run.log tail 16
> Then the full run: three hundred plus five hundred sentences, in three languages. That is two thousand four hundred answers, sent twenty per request.
> The LLM scored {{llm_baseline.json:accuracy.pooled_dev_sub|pct}} on dev sub and {{llm_baseline.json:accuracy.pooled_test_sub|pct}} on test sub, pooled over the three languages. It never returned an invalid label.
> Notice that it is almost equally good in all three languages. That is the bar the router has to meet.

## Phase 3: two real errors and their fixes
- ReadTimeout from the HTTP layer crashed the run: now mapped to a retried error
- WSL stopped mid-run and left zero bytes at the end of the cache file
- Fix: fsync after every write; loader drops crash padding but still fails on real corruption
- A regression test for each fix
visual: code src/router/llm.py 274-298
> Two real errors happened here. First, a read timeout from the HTTP library was not caught by my retry logic, and it crashed the whole run. I mapped all network errors to a retried error, and wrote a test for it.
> Second, the Linux virtual machine stopped in the middle of a run. The cache file ended with two thousand zero bytes, and the next run crashed while reading it.
> The fix was at the root. Every write is now flushed and synced to disk. And the loader removes crash padding, but still fails loudly on any other broken line, so real corruption is never hidden.

## Phase 3: the real cost of one message
- 50 dev_sub sentences, one request each, like production
- Median latency {{llm_single_call.json:latency_ms_median|ms}}, p95 {{llm_single_call.json:latency_ms_p95|ms}}
- About {{llm_single_call.json:mean_input_tokens|int}} input and {{llm_single_call.json:mean_output_tokens|f1}} output tokens per call
- Accuracy single vs batched on the same 50: {{llm_single_call.json:single_call_accuracy|pct}} vs {{llm_single_call.json:batched_accuracy_same_items|pct}}
visual: output logs/03_llm_single.log tail 16
> Batching is good for the free tier, but a real bot sends one message at a time. So I also sent fifty messages one by one.
> The median latency was {{llm_single_call.json:latency_ms_median|ms}}, and the slowest five percent took {{llm_single_call.json:latency_ms_p95|ms}} or more.
> Each call used about {{llm_single_call.json:mean_input_tokens|int}} input tokens, mostly the label list and examples. These real token counts are what I use for the cost math.

## Phase 4: the small model
- MuRIL encoder from Hugging Face, loaded with AutoModel
- Our own head: dropout + nn.Linear(768, 60) on the first token
- Output: 60 scores (logits) per sentence
visual: code src/router/model.py 15-30
> Phase four: the small model. I load the MuRIL encoder, and add my own classification head: dropout, and one linear layer from 768 numbers to 60 intents.
> The first token's vector summarises the sentence. The head turns it into sixty scores, called logits. The highest score is the predicted intent.
> Writing the head myself keeps the PyTorch part clear and easy to explain.

## Phase 4: my own training loop
- AdamW, 10% warmup then linear decay, gradient clipping 1.0
- fp16 mixed precision on the GPU, seed 42
- English train only; early stopping on English dev
- Word embeddings frozen: fits 4 GB and keeps the multilingual vocabulary
visual: code src/router/train.py 142-162
> I did not use the Hugging Face Trainer. I wrote the loop: get a batch, forward pass, cross entropy loss, backward pass, clip the gradients, and an AdamW step.
> The learning rate warms up for the first ten percent of steps, then goes down in a straight line.
> I trained on English only. After every epoch I measure English dev accuracy, keep the best checkpoint, and stop if it does not improve for two epochs.
> The big word embedding table is frozen. It saves GPU memory and keeps the shared multilingual vocabulary intact.

## Phase 4: the first run was not finished learning
- Smoke run: 30 steps, {{smoke_run.json:sec_per_step|f3}} s per step, about {{smoke_run.json:est_total_train_minutes|f1}} min for 5 epochs
- Run 1 (5 epochs): English dev still rising at the end ({{training_log_run1_5epochs.json:best_en_dev_acc|pct}})
- Fix, decided on English dev only: allow up to 15 epochs, let early stopping decide
visual: image results/charts/training_curve_run1_5epochs.png
> Before the full run, a thirty step smoke run estimated the training time: about {{smoke_run.json:est_total_train_minutes|f1}} minutes. Far below my four hour limit, so I kept MuRIL.
> But the first real run had a problem. After five epochs, English dev accuracy was still going up, at {{training_log_run1_5epochs.json:best_en_dev_acc|pct}}. The model had not finished learning.
> So I raised the limit to fifteen epochs and let early stopping on English dev decide. I kept the first run's log, to be honest about it.

## Phase 4: the final training run
- Best epoch: {{training_log.json:best_epoch|int}}
- Best English dev accuracy: {{training_log.json:best_en_dev_acc|pct}}
- Training time on the laptop GPU: {{training_log.json:train_minutes|f1}} minutes
visual: image results/charts/training_curve.png
> Here is the final training curve. The blue line is the training loss going down. The purple line is English dev accuracy.
> The best epoch was {{training_log.json:best_epoch|int}}, with {{training_log.json:best_en_dev_acc|pct}} on English dev. The whole run took {{training_log.json:train_minutes|f1}} minutes on the laptop GPU.

## Phase 4: zero-shot results
- Full test set, English: {{small_model.json:accuracy.en_test_full|pct}}
- Hindi (never trained on Hindi): {{small_model.json:accuracy.hi_test_full|pct}}
- Kannada (never trained on Kannada): {{small_model.json:accuracy.kn_test_full|pct}}
- Published MuRIL zero-shot (IndicXTREME): 82.4% Hindi, 77.5% Kannada
visual: output logs/04_evaluate.log tail 12
> Now the interesting part. On the full English test set the small model scores {{small_model.json:accuracy.en_test_full|pct}}.
> On Hindi, which it never saw in training, it scores {{small_model.json:accuracy.hi_test_full|pct}}. On Kannada, {{small_model.json:accuracy.kn_test_full|pct}}.
> This is zero-shot cross-lingual transfer. MuRIL learned during pre-training that a Hindi sentence and its English translation mean the same thing, so an English classifier partly works in Hindi.
> The published MuRIL numbers are higher, partly because those papers chose checkpoints differently. I only used English dev, to keep it truly zero-shot.

## Phase 5: can we trust the confidence?
- Neural networks are often over-confident
- Temperature scaling: divide the logits by one number T before softmax
- T is fitted on English dev by minimising negative log-likelihood
- T = {{routing.json:temperature|f2}}; ECE on English dev {{routing.json:calibration.ece_en_dev_full.before|f3}} -> {{routing.json:calibration.ece_en_dev_full.after|f3}}
visual: image results/charts/reliability.png
> A threshold only makes sense if the confidence means something. If the model says ninety percent, it should be right about ninety percent of the time.
> Temperature scaling fixes this with one number, T. I divide all the scores by T before the softmax. The predicted intent does not change, only the confidence.
> T was fitted on English dev and came out as {{routing.json:temperature|f2}}. The expected calibration error, called E C E, went from {{routing.json:calibration.ece_en_dev_full.before|f3}} to {{routing.json:calibration.ece_en_dev_full.after|f3}}. Lower is better.

## Phase 5: choosing the threshold, on dev only
- Try t from 0.00 to 1.00 in steps of 0.01, plus 1.01 (everything to the LLM)
- Rule: fewest LLM calls while accuracy >= LLM-only accuracy - 1 point
- On pooled dev_sub, LLM-only = {{routing.json:llm_only_dev_sub_pooled_acc|pct}}, target {{routing.json:target_dev_sub_acc|pct}}
- Chosen t = {{routing.json:chosen_threshold.t|f2}}, sending {{routing.json:chosen_threshold.llm_share|pct}} of dev messages to the LLM
visual: code src/router/route.py 29-39
> Now the threshold. I tried every value from zero to one, in steps of one hundredth, plus one point zero one, which sends everything to the LLM.
> The rule was written down before looking at any test data: pick the threshold with the fewest LLM calls, while the router stays within one percentage point of the LLM-only accuracy on dev sub.
> On dev, the LLM alone scored {{routing.json:llm_only_dev_sub_pooled_acc|pct}}, so the target was {{routing.json:target_dev_sub_acc|pct}}. The chosen threshold is {{routing.json:chosen_threshold.t|f2}}.

## Phase 5: the trade-off curve
- Each line: accuracy vs share of messages sent to the LLM
- Left: dev_sub, used to choose t. Right: test_sub, t fixed
- Dots mark the chosen threshold
visual: image results/charts/accuracy_vs_llm_share.png
> This chart shows the whole trade-off. On the left is dev sub, on the right is test sub. Each line is one language, and the dashed line is all three together.
> Moving right means sending more messages to the LLM. The dots show the chosen threshold.
> Look at the English line. With almost nothing sent to the LLM, it is already high, and a few LLM answers push it even higher. Hindi and Kannada start lower and climb fast as their unsure messages go to the LLM.
> Notice that the test curve peaks around forty percent. The two models make different mistakes. My rule chose the fewest LLM calls instead, because that was the goal I wrote down before seeing any results.

## Phase 5: final results on test_sub
- Small model only: {{routing.json:test_sub.pooled.small_only_acc|pct}}
- LLM only: {{routing.json:test_sub.pooled.llm_only_acc|pct}}
- Router: {{routing.json:test_sub.pooled.router_acc|pct}}, with {{routing.json:test_sub.pooled.llm_share|pct}} sent to the LLM
- Oracle (right if either model is right): {{routing.json:test_sub.pooled.oracle_acc|pct}}
visual: none
> Here are the final numbers on test sub, all three languages together, with the threshold fixed from dev.
> The small model alone: {{routing.json:test_sub.pooled.small_only_acc|pct}}. The LLM alone: {{routing.json:test_sub.pooled.llm_only_acc|pct}}. The router: {{routing.json:test_sub.pooled.router_acc|pct}}, while sending only {{routing.json:test_sub.pooled.llm_share|pct}} of messages to the LLM.
> So the router matches the LLM-only accuracy, with about one LLM call for every nine messages. The accuracy change is {{routing.json:savings_test_sub_pooled.accuracy_change_pp|pp}}.
> The oracle, which is right whenever either model is right, reaches {{routing.json:test_sub.pooled.oracle_acc|pct}}. It shows how much a perfect router could still gain.

## Phase 5: per language
- English: router {{routing.json:test_sub.en.router_acc|pct}}, {{routing.json:test_sub.en.llm_share|pct}} to the LLM
- Hindi: router {{routing.json:test_sub.hi.router_acc|pct}}, {{routing.json:test_sub.hi.llm_share|pct}} to the LLM
- Kannada: router {{routing.json:test_sub.kn.router_acc|pct}}, {{routing.json:test_sub.kn.llm_share|pct}} to the LLM
visual: none
> Per language, the picture makes sense. English is the language the small model was trained on, so it keeps the most messages: only {{routing.json:test_sub.en.llm_share|pct}} go to the LLM.
> Hindi sends {{routing.json:test_sub.hi.llm_share|pct}}, and Kannada sends {{routing.json:test_sub.kn.llm_share|pct}}. The less the small model knows a language, the more often it is unsure, and the router asks for help.

## Phase 5: cost and latency
- LLM cost per 1000 messages: {{routing.json:test_sub.pooled.llm_only_cost_per_1000_usd|usd}} -> {{routing.json:test_sub.pooled.router_cost_per_1000_usd|usd}}
- LLM calls avoided: {{routing.json:savings_test_sub_pooled.llm_calls_avoided_share|pct}}
- Average latency: {{routing.json:test_sub.pooled.llm_only_avg_latency_ms|ms}} -> {{routing.json:test_sub.pooled.router_avg_latency_ms|ms}}
- Accuracy change vs LLM only: {{routing.json:savings_test_sub_pooled.accuracy_change_pp|pp}}
- Small model on the CPU: median {{small_model.json:cpu_latency_single_message.median_ms|ms}} per message
visual: image results/charts/cost_per_1000.png
> Speed matters for a voice bot. On the CPU, with four threads, the small model answers one message in a median of {{small_model.json:cpu_latency_single_message.median_ms|ms}}, against {{llm_single_call.json:latency_ms_median|ms}} for one LLM call.
> Now money and time. Using the paid price of the model and the real token counts, the LLM costs {{routing.json:test_sub.pooled.llm_only_cost_per_1000_usd|usd}} per thousand messages. The router costs {{routing.json:test_sub.pooled.router_cost_per_1000_usd|usd}}.
> The router avoids {{routing.json:savings_test_sub_pooled.llm_calls_avoided_share|pct}} of LLM calls, and the average time per message drops from {{routing.json:test_sub.pooled.llm_only_avg_latency_ms|ms}} to {{routing.json:test_sub.pooled.router_avg_latency_ms|ms}}.
> The accuracy change against the LLM alone is {{routing.json:savings_test_sub_pooled.accuracy_change_pp|pp}}.

## Phase 5: real examples
- Confident messages stay with the small model
- Unsure ones go to the LLM
- Hindi and Kannada drawn with Noto fonts
visual: image results/charts/examples_table.png
> Here are eight real test messages. The top four stayed with the small model because it was confident. The bottom four went to the LLM.
> You can see English, Hindi and Kannada sentences, the true intent, the small model's guess and confidence, and the final answer.

## Phase 6: the Flask API
- GET /health returns status ok
- POST /predict returns intent, confidence, routed_to and latency_ms
- Empty text or more than 500 characters: HTTP 400
- If the LLM fails or times out: keep the small model's answer (small_fallback)
visual: code app/app.py 115-141
> Phase six is the API, built with Flask. The model is loaded once when the app starts, and the threshold is read from the routing results file.
> The predict endpoint validates the input. Empty text, or more than five hundred characters, returns an error 400 with a clear message.
> If the LLM fails or times out, the app never crashes. It returns the small model's answer and marks it small fallback.

## Phase 6: the demo
- One English, one Hindi and one Kannada sentence
- Real responses from the running app
visual: output logs/06_smoke.log
> Here is the smoke test with real requests. One English sentence, one Hindi and one Kannada.
> Each response shows the intent, the confidence, where it was routed, and how long it took.

## Phase 7: serving it like production
- NGINX on 127.0.0.1:8080, the front door
- gunicorn on 127.0.0.1:8000 with 3 worker processes
- preload_app loads the model once before the workers start
- 3 workers x 4 PyTorch threads = 12 cores
visual: image results/charts/serving.png
> Flask's own server is only for development. In production, a request goes from the browser to NGINX, then to gunicorn, then to the Flask app, and then to the model or the LLM.
> Gunicorn runs three worker processes. Each one uses four PyTorch threads, so together they use the twelve cores without fighting over them.
> NGINX sits in front. It handles slow clients, limits the request size, and can serve many connections.

## Phase 7: NGINX without sudo
- Project-local config, started with nginx -p prefix -c nginx.conf -e error log
- pid file and all temp folders inside deploy/nginx/
- No user directive, because it does not run as root
visual: code deploy/nginx/nginx.conf 6-44
> I ran NGINX as a normal user, so no sudo password is needed. The trick is a project-local config.
> The pid file and every temp folder live inside the project, because a normal user cannot write to the system folders. It listens on port 8080 and passes requests to gunicorn.

## Phase 7: the load test
- Locust, headless: {{loadtest.json:users|int}} users for {{loadtest.json:duration|raw}}, small model only, through NGINX
- {{loadtest.json:requests|int}} requests, {{loadtest.json:failures|int}} failures
- {{loadtest.json:requests_per_sec|f1}} requests per second
- p50 {{loadtest.json:p50_ms|ms}}, p95 {{loadtest.json:p95_ms|ms}}
visual: output logs/07_loadtest_summary.log
> Then a load test with Locust. {{loadtest.json:users|int}} simulated users sent real test sentences through NGINX, with the LLM switched off so no API quota was spent.
> The server handled {{loadtest.json:requests_per_sec|f1}} requests per second, with {{loadtest.json:failures|int}} failures. Half of the requests finished within {{loadtest.json:p50_ms|ms}}, and ninety-five percent within {{loadtest.json:p95_ms|ms}}.
> The bottleneck is the CPU. One message alone takes about {{small_model.json:cpu_latency_single_message.median_ms|ms}}, but under load the twelve cores are shared by three workers and the load generator, so requests wait in a queue. To go faster I would quantize the model or add CPU, not add more workers.
> One real bug happened here. The first load test showed zero requests. The workers had inherited a broken PyTorch thread pool from the master process and froze. The fix was one line: keep the master single-threaded, and let each worker set its own threads after it starts.

## Phase 8: the quality gate
- make verify: ruff lint + all pytest tests + number check
- Result tables in the docs are generated from results/ and checked
- Independent review; fresh-clone test
visual: output logs/08_verify.log tail 12
> Phase eight is the quality gate. One command, make verify, runs the linter, every unit test, and a number check.
> The number check is important. Every results table in the readme and the report is generated from the results folder. If anyone edits a number by hand, the check fails.
> Then I had the repo reviewed independently for test leakage, secrets and broken instructions.

## How to explain it in an interview: 30 seconds
- Small multilingual model first, LLM only when unsure
- Trained on English only, works zero-shot on Hindi and Kannada
- Threshold chosen on dev; final numbers on test
- Served with Flask, gunicorn and NGINX, load-tested
visual: none
> If an interviewer asks you to explain the project in thirty seconds, say this.
> I built an intent router for a voice bot. A small multilingual model, fine-tuned in PyTorch on English only, answers when it is confident. Otherwise the message goes to an LLM.
> On Hindi and Kannada test data, the router gets {{routing.json:test_sub.pooled.router_acc|pct}} accuracy while sending only {{routing.json:test_sub.pooled.llm_share|pct}} of messages to the LLM. I served it with Flask, gunicorn and NGINX and load-tested it.

## Interview questions, part 1
- Why route instead of always using the LLM?
- Why choose the threshold on dev and not test?
- Why does English training work for Kannada?
visual: none
> Question one. Why route, instead of always using the LLM? Because most messages are easy. The small model handles them for free in milliseconds, and we pay for the LLM only on the hard ones.
> Question two. Why choose the threshold on dev data? Because if you tune on test data, your test score is no longer an honest estimate. It looks better than reality.
> Question three. Why does English training work for Kannada? MuRIL was pre-trained on many languages together, including translation pairs. So similar meanings get similar vectors in every language, and the English classifier partly carries over.

## Interview questions, part 2
- What does temperature scaling do?
- What is the bottleneck when serving?
- What would you do next?
visual: none
> Question four. What does temperature scaling do? It divides the scores by one number before the softmax, so the confidence matches the real accuracy. It never changes the prediction.
> Question five. What is the bottleneck? CPU-bound model inference. Each worker answers one request at a time, so throughput is limited by the cores, not by NGINX or Flask.
> And what next? Quantize the model to make it faster, add a little Hindi and Kannada training data, and test with real code-mixed speech.

## Summary
- Code, results, docs and this video are all in the repository
- README.md, FINAL_REPORT.md and docs/INTERVIEW_GUIDE.md
- Every number comes from the results folder
visual: image results/charts/architecture.png
> That is the whole project. A small model trained on English, a big model for the hard cases, an honest threshold, and a real server.
> The readme, the final report and the interview guide are in the repository, and every number in them comes from the results folder.
> Thank you for watching.
