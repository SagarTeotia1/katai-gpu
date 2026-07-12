.PHONY: up down logs build shell-backend shell-vllm restart clean test chat analyze parallel help \
        pipeline pipeline-seq pipeline-context pipeline-reindex pipeline-status \
        transcribe transcribe-gpu transcribe-urls analyze-context index-context index-pinecone index-neo4j \
        color-grade export-lut grade-apply emotion-arcs \
        query retrieve edit edit2 direct \
        install install-diarization install-whisper-gpu \
        logs-vllm logs-backend logs-frontend \
        video video-chunk video-fast video-semantic video-bench vision-bench \
        cast-analysis whisper-up whisper-logs whisper-health \
        neo4j-up neo4j-logs list-models status health-backend health-vllm gpu-info \
        render render-dry

ifneq (,$(wildcard .env))
  include .env
  export
endif

COMPOSE            := docker compose
BACKEND_CONTAINER  := katai-backend
VLLM_CONTAINER     := katai-vllm
FRONTEND_CONTAINER := katai-frontend
VLLM_PORT          ?= 8000
BACKEND_PORT       ?= 8080
FRONTEND_PORT      ?= 3000
WHISPER_PORT       ?= 9000
MODEL              ?= $(MODEL_ID)
MODEL              ?= Qwen/Qwen3.6-27B
N                  ?= 4

##@ General

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install all host-side script dependencies (pinecone, neo4j, resemblyzer, sklearn, opencv, faster-whisper, setuptools, ffmpeg...)
	pip install -r scripts/requirements.txt
	pip install faster-whisper setuptools
	conda install -c conda-forge ffmpeg -y 2>/dev/null || apt-get install -y ffmpeg 2>/dev/null || echo "WARNING: ffmpeg not installed — install manually: conda install -c conda-forge ffmpeg"

install-diarization: ## Install speaker diarization deps only (resemblyzer + soundfile + sklearn)
	pip install resemblyzer soundfile scikit-learn

install-whisper-gpu: ## Install faster-whisper for local GPU transcription (no Docker needed)
	pip install faster-whisper

##@ Docker

up: ## Build images and start all services
	@if [ ! -f .env ]; then \
		echo "No .env found — copying from .env.example"; \
		cp .env.example .env; \
	fi
	$(COMPOSE) up --build -d
	@echo ""
	@echo "  Services starting — first run downloads model from HuggingFace (~54 GB)"
	@echo "    vLLM API    → http://localhost:$(VLLM_PORT)"
	@echo "    Backend     → http://localhost:$(BACKEND_PORT)"
	@echo "    Frontend    → http://localhost:$(FRONTEND_PORT)"
	@echo ""
	@echo "  Run 'make logs-vllm' to follow vLLM startup + model load progress."

down: ## Stop and remove containers
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) restart

build: ## Build images without starting
	$(COMPOSE) build

clean: ## Remove containers, images, and volumes (WARNING: deletes model cache)
	$(COMPOSE) down --rmi local --volumes --remove-orphans

##@ Logs

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

logs-vllm: ## Follow vLLM logs
	$(COMPOSE) logs -f vllm

logs-backend: ## Follow backend logs
	$(COMPOSE) logs -f backend

logs-frontend: ## Follow frontend logs
	$(COMPOSE) logs -f frontend

##@ Development

shell-backend: ## Shell into backend container
	docker exec -it $(BACKEND_CONTAINER) /bin/bash

shell-vllm: ## Shell into vLLM container
	docker exec -it $(VLLM_CONTAINER) /bin/bash

##@ Testing

test: ## Full stack health check + test inference
	@echo "=== vLLM health ==="
	@python3 -c "import urllib.request; r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/health', timeout=5); print('OK' if r.status == 200 else 'FAIL')"
	@echo ""
	@echo "=== Backend health ==="
	@python3 -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:$(BACKEND_PORT)/health', timeout=5); print(json.dumps(json.loads(r.read()), indent=2))"
	@echo ""
	@echo "=== Models ==="
	@python3 -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/v1/models', timeout=5); [print(' •', m['id']) for m in json.loads(r.read()).get('data', [])]"
	@echo ""
	@echo "=== Test inference ==="
	@python3 -c "\
import urllib.request, json; \
payload = json.dumps({'model':'$(MODEL)','messages':[{'role':'user','content':'What is 2+2? Answer in one word.'}],'max_tokens':20,'temperature':0.1}).encode(); \
req = urllib.request.Request('http://localhost:$(VLLM_PORT)/v1/chat/completions', data=payload, headers={'Content-Type':'application/json'}); \
data = json.loads(urllib.request.urlopen(req, timeout=60).read()); \
print('Response:', data['choices'][0]['message']['content'].strip()); \
print('Tokens used:', data['usage']['total_tokens'])"

chat: ## Chat with model — usage: make chat MSG="your question"
	@python3 -c "\
import urllib.request, json; \
msg = '$(MSG)' if '$(MSG)' else 'What can you do?'; \
payload = json.dumps({'messages': [{'role':'user','content': msg}], 'max_tokens': 1024}).encode(); \
req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/chat', data=payload, headers={'Content-Type':'application/json'}); \
data = json.loads(urllib.request.urlopen(req, timeout=120).read()); \
print(data['content'])"

video-chunk: ## Parallel chunk analysis — splits video into N chunks, processes concurrently, merges
	@python3 scripts/chunk_analysis.py \
		--vid "$(VID)" \
		--chunks $(N) \
		--backend http://localhost:$(BACKEND_PORT) \
		$(if $(DURATION),--duration "$(DURATION)") \
		$(if $(TRANSCRIPT),--transcript "$(TRANSCRIPT)")

video-fast: ## Fast frame-based chunk analysis — usage: make video-fast VID="https://..." N=4
	@python3 scripts/fast_chunk_analysis.py \
		--vid "$(VID)" \
		--chunks $(N) \
		--backend http://localhost:$(BACKEND_PORT) \
		$(if $(DURATION),--duration "$(DURATION)") \
		$(if $(FPS),--fps "$(FPS)")

video-semantic: ## Full semantic JSON analysis — saves to output/ — usage: make video-semantic VID="https://..."
	@python3 scripts/semantic_analysis.py \
		--vid "$(VID)" \
		--backend http://localhost:$(BACKEND_PORT) \
		$(if $(TRANSCRIPT),--transcript "$(TRANSCRIPT)")

video: ## Analyze video — usage: make video VID="https://..." PROMPT="describe"
	@python3 -c "\
import urllib.request, json; \
vid = '$(VID)'; \
prompt = '$(PROMPT)' if '$(PROMPT)' else 'Analyze this video completely. Describe every scene, action, object, person, text, and detail.'; \
payload = json.dumps({'video_url': vid, 'prompt': prompt}).encode(); \
req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/video/analyze', data=payload, headers={'Content-Type':'application/json'}); \
data = json.loads(urllib.request.urlopen(req, timeout=600).read()); \
print(data['description'])"

video-bench: ## Parallel video benchmark — usage: make video-bench VID1="..." VID2="..." VID3="..." VID4="..."
	@python3 scripts/video_bench.py \
		--backend http://localhost:$(BACKEND_PORT) \
		$(if $(VID1),--vid "$(VID1)") \
		$(if $(VID2),--vid "$(VID2)") \
		$(if $(VID3),--vid "$(VID3)") \
		$(if $(VID4),--vid "$(VID4)")

analyze: ## Analyze image — usage: make analyze IMG="https://..." PROMPT="describe"
	@python3 -c "\
import urllib.request, json; \
img = '$(IMG)'; \
prompt = '$(PROMPT)' if '$(PROMPT)' else 'Describe every object, color, text, and detail in this image.'; \
payload = json.dumps({'image_url': img, 'prompt': prompt}).encode(); \
req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/vision/analyze', data=payload, headers={'Content-Type':'application/json'}); \
data = json.loads(urllib.request.urlopen(req, timeout=300).read()); \
print(data['description'])"

vision-bench: ## Fire all 4 images in parallel — full vision benchmark
	@python3 scripts/vision_bench.py --backend http://localhost:$(BACKEND_PORT)

CAST ?= cast.json

cast-analysis: ## Analyze cast appearance from crop images — usage: make cast-analysis CAST=cast.json
	@python3 scripts/cast_analysis.py $(CAST) --backend http://localhost:$(BACKEND_PORT)

whisper-up: ## Start only the whisper service — usage: make whisper-up
	@$(COMPOSE) up --build -d whisper

whisper-logs: ## Follow whisper service logs
	@$(COMPOSE) logs -f whisper

whisper-health: ## Check whisper service health
	@curl -s http://localhost:$(WHISPER_PORT)/health | python3 -m json.tool

WHISPER_WORKERS ?= 0
WHISPER_MODEL   ?= large-v3
WHISPER_DEVICE  ?= cuda
WHISPER_COMPUTE ?= float16

transcribe: ## Transcribe videos via Docker whisper service — usage: make transcribe CAST=cast.json [WHISPER_WORKERS=0=auto]
	@python3 scripts/transcribe.py --cast $(CAST) \
		--whisper http://localhost:$(WHISPER_PORT) \
		--backend http://localhost:$(BACKEND_PORT) \
		--workers $(WHISPER_WORKERS)

transcribe-gpu: ## Transcribe on host GPU (faster-whisper, no Docker) — usage: make transcribe-gpu CAST=cast.json [WHISPER_MODEL=large-v3]
	@python3 scripts/transcribe.py --cast $(CAST) \
		--local \
		--model $(WHISPER_MODEL) \
		--device $(WHISPER_DEVICE) \
		--compute-type $(WHISPER_COMPUTE)

transcribe-urls: ## Transcribe raw video URLs — usage: make transcribe-urls VIDS="url1 url2"
	@python3 scripts/transcribe.py --videos $(VIDS) --whisper http://localhost:$(WHISPER_PORT)

WORKERS ?= 24
CHUNKS  ?= 8
PLANNER ?= semantic

analyze-context: ## Full semantic video context — usage: make analyze-context CAST=cast.json [PLANNER=semantic|scene|fixed] [CHUNKS=8]
	@python3 scripts/analyze_context.py --cast $(CAST) \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		--backend http://localhost:$(BACKEND_PORT) \
		--workers $(WORKERS) \
		--chunks $(CHUNKS) \
		--planner $(PLANNER)

pipeline: ## ONE CMD — cast→transcript→diarize→context→index — usage: make pipeline CAST=cast.json [CHUNKS=8] [WORKERS=24] [PLANNER=semantic]
	@python3 scripts/pipeline.py $(CAST) \
		--backend http://localhost:$(BACKEND_PORT) \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		--whisper http://localhost:$(WHISPER_PORT) \
		--chunks $(CHUNKS) \
		--workers $(WORKERS) \
		--whisper-workers $(WHISPER_WORKERS) \
		--planner $(PLANNER)

pipeline-seq: ## Sequential GPU pipeline: Whisper(GPU)→Cast→Context — usage: make pipeline-seq CAST=cast.json [WHISPER_MODEL=large-v3]
	@python3 scripts/pipeline.py $(CAST) \
		--sequential \
		--local-whisper \
		--whisper-model $(WHISPER_MODEL) \
		--whisper-device $(WHISPER_DEVICE) \
		--whisper-compute-type $(WHISPER_COMPUTE) \
		--backend http://localhost:$(BACKEND_PORT) \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		--chunks $(CHUNKS) \
		--workers $(WORKERS) \
		--planner $(PLANNER)

pipeline-context: ## Skip cast+transcribe, run context+index only — usage: make pipeline-context CAST=cast.json
	@python3 scripts/pipeline.py $(CAST) \
		--skip-cast auto --skip-transcribe auto \
		--backend http://localhost:$(BACKEND_PORT) \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		--whisper http://localhost:$(WHISPER_PORT) \
		--chunks $(CHUNKS) \
		--workers $(WORKERS) \
		--planner $(PLANNER)

pipeline-reindex: ## Re-run indexing only (skip all analysis) — usage: make pipeline-reindex CAST=cast.json
	@python3 scripts/pipeline.py $(CAST) \
		--skip-cast auto --skip-transcribe auto --skip-context \
		--backend http://localhost:$(BACKEND_PORT) \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		--whisper http://localhost:$(WHISPER_PORT)

pipeline-status: ## Show last pipeline run summary JSON
	@python3 -c "\
import json, glob, sys; \
files = sorted(glob.glob('output/pipeline_*.json')); \
f = files[-1] if files else None; \
(print('No pipeline runs found in output/') or sys.exit(1)) if not f else \
print(json.dumps(json.loads(open(f).read()), indent=2))"

index-context: ## Index context JSONs → Pinecone + Neo4j — usage: make index-context [FILES="output/context_*.json"]
	@python3 scripts/index_context.py $(FILES)

index-pinecone: ## Index to Pinecone only (skip Neo4j)
	@python3 scripts/index_context.py --no-neo4j $(FILES)

index-neo4j: ## Index to Neo4j only (skip Pinecone)
	@python3 scripts/index_context.py --no-pinecone $(FILES)

color-grade: ## Generate per-scene color grade plan — usage: make color-grade CAST=cast.json [STYLE="Netflix documentary"]
	@python3 - <<-'EOF'
	import json, sys, os, glob
	from pathlib import Path
	style = os.environ.get("STYLE", "cinematic")
	files = sorted(glob.glob("output/context_*.json"), key=os.path.getmtime)
	if not files:
	    print("ERROR: no context_*.json in output/ — run make analyze-context first"); sys.exit(1)
	sys.path.insert(0, "scripts")
	import analyze_context as ac
	import urllib.request, json as _json
	for f in files:
	    ctx = json.loads(Path(f).read_text(encoding="utf-8"))
	    prompt = ac.build_color_grade_prompt(ctx, style_request=style)
	    payload = {"model": ac.MODEL_ID, "messages": [{"role":"system","content":prompt},{"role":"user","content":"/no_think\n\nGenerate the color grade plan."}], "max_tokens":4096, "temperature":0.0, "response_format":{"type":"json_object"}, "chat_template_kwargs":{"enable_thinking":False}}
	    req = urllib.request.Request(f"http://localhost:$(VLLM_PORT)/v1/chat/completions", data=_json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
	    resp = _json.loads(urllib.request.urlopen(req, timeout=300).read())
	    raw = resp["choices"][0]["message"].get("content","")
	    out = Path("output") / f"color_grade_{Path(f).stem}_{ac.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
	    out.write_text(raw, encoding="utf-8")
	    print(f"  [{ctx.get('video_id','')}] color grade -> {out}")
	EOF

export-lut: ## Export .cube LUT for each color chunk — usage: make export-lut [SIZE=17]
	@python3 - <<-'EOF'
	import json, sys, glob, os
	from pathlib import Path
	sys.path.insert(0, "scripts")
	import color_analyzer as ca
	size = int(os.environ.get("SIZE", "17"))
	files = sorted(glob.glob("output/context_*.json"), key=os.path.getmtime)
	if not files:
	    print("ERROR: no context_*.json in output/ — run make analyze-context first"); sys.exit(1)
	for f in files:
	    ctx = json.loads(Path(f).read_text(encoding="utf-8"))
	    vid = ctx.get("video_id", "unknown")
	    for i, ct in enumerate(ctx.get("color_timeline", [])):
	        grade = ct.get("grade") or {}
	        if not grade.get("grade_needed", False):
	            continue
	        slug = f"{vid}_chunk{i:02d}_{ct.get('look','')}"
	        out  = str(Path("output") / f"grade_{slug}.cube")
	        lut_path = ca.export_lut(grade, lut_size=size, out_path=out, title=slug)
	        print(f"  [{vid} chunk{i}] LUT -> {lut_path}  filter: {ca.build_ffmpeg_filter(grade)}")
	EOF

grade-apply: ## Print ready ffmpeg commands to apply color grade — usage: make grade-apply VID="input.mp4"
	@python3 - <<-'EOF'
	import json, sys, glob, os
	from pathlib import Path
	sys.path.insert(0, "scripts")
	import color_analyzer as ca
	vid_input = os.environ.get("VID", "")
	if not vid_input:
	    print("ERROR: specify VID=input.mp4"); sys.exit(1)
	files = sorted(glob.glob("output/context_*.json"), key=os.path.getmtime)
	if not files:
	    print("ERROR: no context_*.json in output/ — run make analyze-context first"); sys.exit(1)
	ctx = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
	color_tl = ctx.get("color_timeline", [])
	if not color_tl:
	    print("No color_timeline found — re-run make analyze-context with opencv installed"); sys.exit(1)
	print(f"\n# Color grade ffmpeg commands for: {ctx.get('video_id', 'unknown')}")
	print(f"# Input: {vid_input}\n")
	for i, ct in enumerate(color_tl):
	    grade = ct.get("grade") or {}
	    vf = ca.build_ffmpeg_filter(grade)
	    t_start = ct.get("start", 0)
	    t_end   = ct.get("end", 0)
	    look    = ct.get("look", "unknown")
	    needed  = grade.get("grade_needed", False)
	    flag    = "GRADE NEEDED" if needed else "ok"
	    print(f"# Chunk {i}: {t_start:.1f}s-{t_end:.1f}s  look={look}  [{flag}]")
	    if needed:
	        stem = Path(vid_input).stem
	        print(f'ffmpeg -ss {t_start:.1f} -to {t_end:.1f} -i "{vid_input}" -vf "{vf}" -c:a copy output/graded_{stem}_chunk{i:02d}.mp4')
	    print()
	EOF

emotion-arcs: ## Print per-speaker emotion intensity arcs — usage: make emotion-arcs [FILES="output/context_*.json"]
	@python3 - <<-'EOF'
	import json, sys, glob
	from pathlib import Path
	files = sorted(glob.glob("output/context_*.json"), key=lambda p: Path(p).stat().st_mtime)
	if not files:
	    print("ERROR: no context_*.json in output/"); sys.exit(1)
	for f in files:
	    ctx  = json.loads(Path(f).read_text())
	    vid  = ctx.get("video_id", "unknown")
	    arcs = ctx.get("emotion_arcs") or {}
	    if not arcs:
	        print(f"[{vid}] No emotion_arcs — re-index after re-running analyze-context"); continue
	    print(f"\n[{vid}] Per-speaker emotion intensity (30s windows):")
	    for pid, windows in sorted(arcs.items()):
	        if not windows:
	            print(f"  {pid}: no windows"); continue
	        peak = max(windows, key=lambda w: w.get("peak_intensity", 0))
	        print(f"  {pid}: peak at {peak['t_start']:.0f}s (intensity={peak['peak_intensity']:.2f}, laughs={peak['laugh_count']})")
	        bar = ""
	        for w in windows:
	            lvl = w.get("mean_intensity", 0)
	            bar += "█" if lvl > 0.7 else "▓" if lvl > 0.4 else "░" if lvl > 0.1 else " "
	        print(f"    arc: [{bar}] (each char = 30s)")
	EOF

query: ## Natural language Q&A over indexed video — usage: make query Q="who laughed hardest and when"
	@python3 scripts/query_context.py "$(Q)" --vllm http://localhost:$(VLLM_PORT)/v1/chat/completions

retrieve: ## RAG — retrieve ranked editable events (no LLM synthesis) — usage: make retrieve Q="best hook moment" [VIDEO=vid1] [TOPK=15] [MIN_SCORE=0.3]
	@python3 scripts/retriever.py \
		--query "$(Q)" \
		$(if $(VIDEO),--video "$(VIDEO)") \
		$(if $(TOPK),--top-k $(TOPK)) \
		$(if $(MIN_SCORE),--min-score $(MIN_SCORE)) \
		$(if $(OUT),--out "$(OUT)")

edit2: ## Lightweight Director+Editor pipeline → compact Edit DSL JSON — usage: make edit2 PROMPT="60s energetic Short" [VIDEO=vid1] [TOPK=20]
	@python3 scripts/editor.py \
		--prompt "$(PROMPT)" \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		$(if $(VIDEO),--video "$(VIDEO)") \
		$(if $(TOPK),--top-k $(TOPK)) \
		$(if $(OUT),--out "$(OUT)")

direct: ## Director+Editor brain → EDL JSON — usage: make direct PROMPT="60s YouTube short of funniest moment" [VIDEO=video1]
	@python3 scripts/director_brain.py "$(PROMPT)" \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		$(if $(VIDEO),--video "$(VIDEO)") \
		$(if $(TOPK),--top-k $(TOPK)) \
		$(if $(MAX_TOKENS),--max-tokens $(MAX_TOKENS))

edit: ## Chief Editor (best) — 3-layer pipeline → primitive-op edit plan — usage: make edit PROMPT="45s Short" [VIDEO=video1] [SAVE_INT=1]
	@python3 scripts/chief_editor.py "$(PROMPT)" \
		--vllm http://localhost:$(VLLM_PORT)/v1/chat/completions \
		$(if $(VIDEO),--video "$(VIDEO)") \
		$(if $(TOPK),--top-k $(TOPK)) \
		$(if $(SAVE_INT),--save-intermediate)

neo4j-up: ## Start Neo4j standalone
	@$(COMPOSE) up -d neo4j

neo4j-logs: ## Follow Neo4j logs
	@$(COMPOSE) logs -f neo4j

parallel: ## Fire N concurrent requests to test vLLM concurrency — usage: make parallel N=8 IMG="https://..."
	@python3 -c "\
import urllib.request, json, time; \
from concurrent.futures import ThreadPoolExecutor, as_completed; \
n = $(N); \
img = '$(IMG)'; \
use_vision = bool(img); \
questions = [ \
  'Explain attention mechanism in transformers in 2 sentences.', \
  'What is gradient descent? Keep it brief.', \
  'What is a neural network? One paragraph.', \
  'Explain backpropagation simply.', \
  'What is the difference between RNN and LSTM?', \
  'What is batch normalization used for?', \
  'Explain dropout regularization.', \
  'What is the vanishing gradient problem?', \
  'What are residual connections in deep learning?', \
  'Explain the encoder-decoder architecture.', \
]; \
def call_chat(i): \
  t0 = time.time(); \
  payload = json.dumps({'messages':[{'role':'user','content': questions[i % len(questions)]}],'max_tokens':200}).encode(); \
  req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/chat', data=payload, headers={'Content-Type':'application/json'}); \
  data = json.loads(urllib.request.urlopen(req, timeout=180).read()); \
  return i, time.time()-t0, data['content'][:120]; \
def call_vision(i): \
  t0 = time.time(); \
  payload = json.dumps({'image_url': img, 'prompt': 'Briefly describe this image in 2 sentences.'}).encode(); \
  req = urllib.request.Request('http://localhost:$(BACKEND_PORT)/api/vision/analyze', data=payload, headers={'Content-Type':'application/json'}); \
  data = json.loads(urllib.request.urlopen(req, timeout=300).read()); \
  return i, time.time()-t0, data['description'][:120]; \
fn = call_vision if use_vision else call_chat; \
print(f'Firing {n} concurrent requests ({\"vision\" if use_vision else \"chat\"})...'); \
print('-' * 60); \
t_start = time.time(); \
with ThreadPoolExecutor(max_workers=n) as ex: \
  futures = [ex.submit(fn, i) for i in range(n)]; \
  for f in as_completed(futures): \
    i, elapsed, preview = f.result(); \
    print(f'[req {i+1:02d}] {elapsed:.1f}s | {preview}...'); \
print('-' * 60); \
print(f'All {n} done in {time.time()-t_start:.1f}s total')"

##@ Render (L6 Executor)

render: ## L6 Executor — editplan JSON → rendered MP4  (PLAN=path/to/editplan.json CAST=path/to/cast.json COLOR=1)
	@python3 scripts/executor.py \
		$(if $(PLAN),$(PLAN)) \
		$(if $(CAST),--cast "$(CAST)") \
		$(if $(COLOR),--apply-color) \
		$(if $(OUT),--output "$(OUT)")

render-dry: ## Dry-run — print ffmpeg commands without executing
	@python3 scripts/executor.py \
		$(if $(PLAN),$(PLAN)) \
		$(if $(CAST),--cast "$(CAST)") \
		$(if $(COLOR),--apply-color) \
		$(if $(OUT),--output "$(OUT)") \
		--dry-run

##@ Model

list-models: ## List models in vLLM
	@python3 -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/v1/models', timeout=5); [print(m['id']) for m in json.loads(r.read()).get('data', [])]"

##@ Health

status: ## Show container status
	$(COMPOSE) ps

health-backend: ## Check backend health
	@python3 -c "import urllib.request, json; r = urllib.request.urlopen('http://localhost:$(BACKEND_PORT)/health', timeout=5); print(json.dumps(json.loads(r.read()), indent=2))"

health-vllm: ## Check vLLM health
	@python3 -c "import urllib.request; r = urllib.request.urlopen('http://localhost:$(VLLM_PORT)/health', timeout=5); print(r.status, r.reason)"

gpu-info: ## Show GPU info
	@docker exec $(VLLM_CONTAINER) nvidia-smi
