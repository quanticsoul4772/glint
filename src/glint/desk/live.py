"""Bank Desk live mode: run one banking task twice at once, Glimmer alone and Glimmer with the Glint tools.

Each pane builds its own τ² environment, agent and user (τ²'s build_environment / build_agent / build_user), and only
the tools pane's environment gets the decision tools (glint.assist.tau2_tools.assist_environment), so the two panes
never share a toolkit or a database. Both run in background threads; `snapshot()` reads each orchestrator's growing
trajectory and returns the same events replay mode shows, then the outcome and the overrule marks when a pane ends.

Glimmer is served by glint.glimmer_serve on Modal (GLINT_GLIMMER_URL, MODAL_PROXY_KEY, MODAL_PROXY_SECRET); the
customer is Muse Spark on Meta's Model API (MODEL_API_KEY). Glimmer calls go through glint.atem; Meta spend is capped.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

from glint.desk.events import outcome, timeline
from glint.meter import BudgetExceeded

SPARK = "openai/muse-spark-1.3-contributor"
DOMAIN = "banking_knowledge"
META_CAP_USD = 1.0  # per desk process; a live pair costs a few cents of Meta credit


@dataclass
class Pane:
    name: str
    orchestrator: object
    thread: threading.Thread | None = None
    result: object = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None


class LiveSession:
    """One live pair. `build(pane_name)` returns (orchestrator, run) where run(orchestrator) returns the finished
    τ² SimulationRun; `label(sim_dict)` returns {message_index: decision} for the overrule marks. `wake()` blocks
    until the Glimmer endpoint is ready (raising if it cannot start); `on_done(session)` runs once both panes end."""

    def __init__(self, task_id: str, build, label=None, wake=None, on_done=None):
        self.id = uuid.uuid4().hex[:12]
        self.task_id = task_id
        self.label, self._wake, self._on_done = label, wake, on_done
        self.started = time.time()
        self.ended: float | None = None
        self.waking = wake is not None
        self.wake_error: str | None = None
        self.panes: dict[str, Pane] = {}
        self._runners = {}
        for name in ("alone", "tools"):
            orchestrator, run = build(name)
            self.panes[name] = Pane(name, orchestrator)
            self._runners[name] = run

    def start(self) -> LiveSession:
        threading.Thread(target=self._control, daemon=True).start()
        return self

    def _control(self) -> None:
        try:
            if self._wake is not None:
                try:
                    self._wake()
                except Exception as e:  # noqa: BLE001  (shown in both panes and the session log)
                    self.wake_error = f"the Glimmer endpoint did not start: {type(e).__name__}: {e}"
                    traceback.print_exc()
                    for pane in self.panes.values():
                        pane.error, pane.finished = self.wake_error, time.time()
                    return
                finally:
                    self.waking = False
            for name, pane in self.panes.items():
                pane.started = time.time()
                pane.thread = threading.Thread(target=self._run, args=(pane, self._runners[name]), daemon=True)
                pane.thread.start()
            for pane in self.panes.values():
                pane.thread.join()
        finally:
            self.ended = time.time()
            if self._on_done is not None:
                self._on_done(self)

    def _run(self, pane: Pane, run) -> None:
        try:
            pane.result = run(pane.orchestrator)
        # An exception in this thread would vanish; it is shown in the pane and printed in full to the server log,
        # and the other pane keeps running
        except Exception as e:  # noqa: BLE001
            pane.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            pane.finished = time.time()

    @property
    def done(self) -> bool:
        return self.ended is not None

    def snapshot(self) -> dict:
        out = {"id": self.id, "task_id": self.task_id, "done": self.done}
        for name, pane in self.panes.items():
            messages = [m.model_dump() for m in list(getattr(pane.orchestrator, "trajectory", []) or [])]
            view = {"status": "error" if pane.error else ("done" if pane.finished else
                                                           ("waking" if self.waking else "running")),
                    "elapsed_s": round((pane.finished or time.time()) - pane.started), "error": pane.error}
            overrules = {}
            if pane.result is not None:
                sim = pane.result.model_dump()
                messages = sim["messages"]
                view["outcome"] = outcome(sim)
                if name == "alone" and self.label is not None:
                    overrules = self.label(sim)
                    view["overruled_writes"] = len(overrules)
            view["events"] = timeline(messages, overrules)
            out[name] = view
        return out


SCALEDOWN_S = 120.0  # glimmer_serve's scaledown_window: the container stays up this long after the last request


class DeskBudget:
    """GPU and usage limits for live mode, persisted in the meter ledger so a restart cannot reset them.

    A live pair keeps the Glimmer container up for its whole run plus the scale-down window, so each finished pair
    records (wall time + SCALEDOWN_S) at the A100 rate: an upper bound, and double-counted if a batch session keeps
    the container up at the same time. Starts are refused once the day's GPU cap or pair count is reached."""

    def __init__(self, gpu_usd_per_day: float, pairs_per_day: int, ledger=None, log_path=None):
        from glint.meter import DEFAULT_LEDGER

        self.gpu_usd_per_day, self.pairs_per_day = gpu_usd_per_day, pairs_per_day
        self.ledger = ledger or DEFAULT_LEDGER
        self.log_path = log_path

    def _meter(self):
        from glint.meter import Caps, Meter

        return Meter(Caps(meta_usd=0.0, gpu_usd=self.gpu_usd_per_day), ledger=self.ledger,
                     phase=f"desk-gpu-{time.strftime('%Y%m%d')}")

    def pairs_today(self) -> int:
        if not self.log_path or not os.path.exists(self.log_path):
            return 0
        today = time.strftime("%Y-%m-%d")
        with open(self.log_path, encoding="utf-8") as fh:
            return sum(1 for line in fh if f'"day": "{today}"' in line)

    def check(self) -> None:
        self._meter().check_gpu()
        if self.pairs_today() >= self.pairs_per_day:
            raise BudgetExceeded(f"the daily limit of {self.pairs_per_day} live runs is reached")

    def record(self, session: LiveSession) -> None:
        import json

        from glint.glimmer_serve import GPU, GPU_USD_PER_HOUR

        seconds = (session.ended or time.time()) - session.started + SCALEDOWN_S
        usd = self._meter().record_gpu(seconds, GPU_USD_PER_HOUR, GPU, tag=f"desk {session.id} {session.task_id}")
        if self.log_path:
            snap = session.snapshot()
            row = {"day": time.strftime("%Y-%m-%d"), "ts": time.time(), "session": session.id,
                   "task_id": session.task_id, "gpu_seconds": round(seconds), "gpu_usd": round(usd, 4),
                   "wake_error": session.wake_error,
                   **{f"{n}_outcome": (snap[n].get("outcome") or {}).get("passed") for n in ("alone", "tools")},
                   **{f"{n}_error": snap[n].get("error") for n in ("alone", "tools")},
                   **{f"{n}_seconds": snap[n]["elapsed_s"] for n in ("alone", "tools")}}
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


def tau2_builder(task, glimmer_args: dict, glimmer_model: str, user_args: dict, seed: int):
    """The real build function: a τ² orchestrator per pane, the tools pane's environment given the decision tools."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.build import _build_env_kwargs, build_agent, build_environment, build_user
    from tau2.runner.simulation import run_simulation

    from glint.assist.tau2_tools import assist_environment

    config = TextRunConfig(domain=DOMAIN, agent="llm_agent", llm_agent=glimmer_model,
                           llm_args_agent=glimmer_args | {"max_tokens": 8192}, user="user_simulator",
                           llm_user=SPARK, llm_args_user=user_args, retrieval_config="bm25_grep", seed=seed)
    env_kwargs = _build_env_kwargs(config, task)

    def build(pane: str):
        env = build_environment(DOMAIN, solo_mode=False, env_kwargs=env_kwargs)
        if pane == "tools":
            assist_environment(env, DOMAIN)
        agent = build_agent(config.effective_agent, env, llm=config.llm_agent, llm_args=config.llm_args_agent,
                            task=task, solo_mode=False)
        user = build_user(config.effective_user, env, task, llm=config.llm_user, llm_args=config.llm_args_user,
                          solo_mode=False)
        orchestrator = Orchestrator(domain=DOMAIN, agent=agent, user=user, environment=env, task=task,
                                    max_steps=config.effective_max_steps, max_errors=config.max_errors, seed=seed,
                                    solo_mode=False, simulation_id=str(uuid.uuid4()),
                                    validate_communication=config.enforce_communication_protocol,
                                    timeout=config.timeout)
        return orchestrator, lambda o: run_simulation(o, env_kwargs=env_kwargs or None)

    return build


def tau2_labeler(task):
    from tau2.domains.banking_knowledge.environment import get_environment

    from glint.decisions import label_run

    def label(sim: dict) -> dict:
        return {d.message_index: {"tool": d.tool, "reasons": d.reasons}
                for d in label_run(sim, task, get_environment(retrieval_variant="bm25_grep"))
                if d.label == "overruled" and d.executed}

    return label


_configured = {}


def configure() -> dict:
    """Route Glimmer through glint.atem and cap Meta spend, once per process. Returns the Glimmer settings, or raises
    KeyError naming the missing environment variable."""
    if _configured:
        return _configured
    import litellm
    from tau2.utils import llm_utils

    from glint import glimmer_serve
    from glint.atem import reparsing
    from glint.meter import Caps, Meter, guarded, litellm_callback

    url, key, secret = os.environ["GLINT_GLIMMER_URL"], os.environ["MODAL_PROXY_KEY"], os.environ["MODAL_PROXY_SECRET"]
    meta = Meter(Caps(meta_usd=META_CAP_USD, gpu_usd=0.0), phase=f"desk-live-{time.strftime('%Y%m%d')}")
    litellm.callbacks = [litellm_callback((meta, ("muse-spark",)))]
    base, glimmer = llm_utils.completion, reparsing(llm_utils.completion)

    def route(*args, **kwargs):
        return (glimmer if "Muse-Glimmer" in kwargs.get("model", "") else base)(*args, **kwargs)

    llm_utils.completion = guarded(route, meta, only=("muse-spark",))
    _configured.update(url=url, glimmer_model=f"openai/{glimmer_serve.MODEL_NAME}",
                       glimmer_args=glimmer_serve.litellm_args(url, key, secret, "medium"),
                       user_args={"api_base": "https://api.meta.ai/v1", "api_key": os.environ["MODEL_API_KEY"],
                                  "temperature": 1.0})
    return _configured
