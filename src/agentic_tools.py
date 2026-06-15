import json
import re
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import pipeline
from langchain.tools import tool
from langchain.agents import create_agent


def _extract_text(result) -> str:
    """Extract the latest text response from a LangChain agent result."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            joined = " ".join(parts).strip()
            if joined:
                return joined
    return ""


def make_executor(agent):
    """Create a simple invoke-compatible executor object without classes."""

    def invoke(payload: dict):
        user_input = payload.get("input", "") if isinstance(payload, dict) else str(payload)
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        return {
            "output": _extract_text(result),
            "intermediate_steps": [],
            "messages": result.get("messages", []) if isinstance(result, dict) else [],
        }

    return SimpleNamespace(invoke=invoke)


def make_memory_executor(agent):
    """Create a stateful invoke-compatible executor object without classes."""
    history = []

    def invoke(payload: dict):
        user_input = payload.get("input", "") if isinstance(payload, dict) else str(payload)
        history_text = "\n".join(history[-8:])
        composed = (
            "Conversation history:\n"
            f"{history_text}\n\n"
            "Current user request:\n"
            f"{user_input}"
        )
        result = agent.invoke({"messages": [{"role": "user", "content": composed}]})
        output = _extract_text(result)
        history.append(f"User: {user_input}")
        history.append(f"Assistant: {output}")
        return {
            "output": output,
            "intermediate_steps": [],
            "messages": result.get("messages", []) if isinstance(result, dict) else [],
        }

    return SimpleNamespace(invoke=invoke)


def build_agent_tools(transformer_runs, preprocess_pipeline_fn, label_map, max_length, device):
    """Create model registry and tool functions for agentic inference."""
    model_registry = {
        name: {
            "checkpoint": run["checkpoint"],
            "classifier": run["classifier"],
            "metrics": run["metrics"],
        }
        for name, run in transformer_runs.items()
    }
    model_name_lookup = {name.lower(): name for name in model_registry}
    best_model_name = max(
        model_registry,
        key=lambda name: model_registry[name]["metrics"].get("cv_f1_macro_mean", -1),
    )

    @lru_cache(maxsize=None)
    def get_feature_extractor(model_checkpoint: str):
        return pipeline(
            "feature-extraction",
            model=model_checkpoint,
            tokenizer=model_checkpoint,
            batch_size=1,
            device=device,
        )

    def embed_tweet_for_model(text: str, model_checkpoint: str):
        processed_text = preprocess_pipeline_fn(
            pd.Series([text]),
            mode="transformer",
        ).iloc[0]

        outputs = get_feature_extractor(model_checkpoint)(
            [processed_text],
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        arr = np.asarray(outputs[0], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr[0].reshape(1, -1), processed_text

    def run_model_prediction(experiment_name: str, text: str):
        config = model_registry[experiment_name]
        classifier = config["classifier"]
        embedding, processed = embed_tweet_for_model(text, config["checkpoint"])

        pred_id = int(classifier.predict(embedding)[0])
        probs = classifier.predict_proba(embedding)[0]
        probabilities = {
            label_map[int(label_id)]: round(float(prob), 4)
            for label_id, prob in zip(classifier.classes_, probs)
        }
        ranking = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        return {
            "model": experiment_name,
            "checkpoint": config["checkpoint"],
            "preprocessed_text": processed,
            "predicted_label": label_map[pred_id],
            "confidence": round(ranking[0][1], 4),
            "probabilities": probabilities,
        }

    def format_prediction_result(result: dict) -> str:
        prob_str = ", ".join(
            f"{label}={prob:.4f}" for label, prob in result["probabilities"].items()
        )
        return (
            f"model: {result['model']}\n"
            f"predicted_label: {result['predicted_label']}\n"
            f"confidence: {result['confidence']:.4f}\n"
            f"probabilities: {prob_str}\n"
            f"preprocessed_text: {result['preprocessed_text']}"
        )

    @tool
    def inspect_tweet_profile(text: str) -> str:
        """Inspect a tweet and summarize routing-relevant signals."""
        tweet = str(text)
        words = tweet.split()
        url_count = len(re.findall(r"http\S+|www\.\S+", tweet))
        mention_count = len(re.findall(r"@\w+", tweet))
        hashtag_count = len(re.findall(r"#\w+", tweet))
        cashtag_count = len(re.findall(r"\$[A-Za-z]+", tweet))
        uppercase_tokens = sum(1 for token in words if token.isupper() and len(token) > 1)
        noise_score = url_count + mention_count + hashtag_count + cashtag_count + uppercase_tokens

        if noise_score >= 2 or len(words) <= 6:
            suggested_strategy = "compare_both_models"
        elif hashtag_count + cashtag_count + mention_count + url_count >= 1:
            suggested_strategy = "prefer_twitter_roberta"
        else:
            suggested_strategy = "prefer_distilbert"

        summary = {
            "char_length": len(tweet),
            "word_count": len(words),
            "urls": url_count,
            "mentions": mention_count,
            "hashtags": hashtag_count,
            "cashtags": cashtag_count,
            "uppercase_tokens": uppercase_tokens,
            "suggested_strategy": suggested_strategy,
        }
        return json.dumps(summary, indent=2)

    @tool
    def get_validation_metrics(model_name: str = "all") -> str:
        """Return metrics for one model or all available models."""
        query = str(model_name).strip()
        if query.lower() == "all":
            rows = []
            ranked_models = sorted(
                model_registry.values(),
                key=lambda item: item["metrics"].get("cv_f1_macro_mean", 0),
                reverse=True,
            )
            for item in ranked_models:
                m = item["metrics"]
                rows.append(
                    f"{m['model']} | cv_f1_macro_mean={m.get('cv_f1_macro_mean', 0):.4f} | "
                    f"cv_f1_macro_std={m.get('cv_f1_macro_std', 0):.4f} | "
                    f"cv_accuracy_mean={m.get('cv_accuracy_mean', 0):.4f}"
                )
            return "\n".join(rows)

        matched = model_name_lookup.get(query.lower())
        if matched is None:
            available = ", ".join(model_registry)
            return f"Unknown model: {model_name}. Available models: {available}"

        m = model_registry[matched]["metrics"]
        return (
            f"{m['model']} | cv_f1_macro_mean={m.get('cv_f1_macro_mean', 0):.4f} | "
            f"cv_f1_macro_std={m.get('cv_f1_macro_std', 0):.4f} | "
            f"cv_accuracy_mean={m.get('cv_accuracy_mean', 0):.4f}"
        )

    @tool
    def classify_with_best_model(text: str) -> str:
        """Classify tweet with the best available model by CV F1-macro."""
        result = run_model_prediction(best_model_name, text)
        return format_prediction_result(result)

    return {
        "model_registry": model_registry,
        "model_name_lookup": model_name_lookup,
        "best_model_name": best_model_name,
        "inspect_tweet_profile": inspect_tweet_profile,
        "get_validation_metrics": get_validation_metrics,
        "classify_with_best_model": classify_with_best_model,
    }


def create_specialist_executors(llm, tools_bundle):
    """Create routing and classification specialist executors."""
    routing_tools = [
        tools_bundle["inspect_tweet_profile"],
        tools_bundle["get_validation_metrics"],
    ]

    classification_tools = [
        tools_bundle["classify_with_best_model"],
    ]

    routing_agent = create_agent(
        model=llm,
        tools=routing_tools,
        system_prompt=(
            "You are a tweet-routing specialist. Inspect tweet profile and use validation metrics to guide decisions. "
            "Recommend using the best validated model for final classification."
        ),
    )
    routing_executor = make_executor(routing_agent)

    classification_agent = create_agent(
        model=llm,
        tools=classification_tools,
        system_prompt=(
            "You are a tweet sentiment classification specialist. "
            "Use classify_with_best_model to produce a grounded sentiment prediction. "
            "Always report final label, confidence, and model name."
        ),
    )
    classification_executor = make_executor(classification_agent)

    return routing_executor, classification_executor, routing_tools, classification_tools


def create_orchestrator_executor(llm, routing_executor, classification_executor):
    """Create orchestrator that calls routing then classification experts."""
    @tool
    def routing_expert(query: str) -> str:
        """Use first for tweet classification/model selection tasks."""
        return routing_executor.invoke({"input": query})["output"]

    @tool
    def classification_expert(query: str) -> str:
        """Run classifier(s) and return grounded final sentiment prediction."""
        return classification_executor.invoke({"input": query})["output"]

    orchestrator_tools = [routing_expert, classification_expert]

    orchestrator_agent = create_agent(
        model=llm,
        tools=orchestrator_tools,
        system_prompt=(
            "You are the orchestrator for tweet classification. "
            "For classification tasks: call routing_expert first, then classification_expert. "
            "For recommendation/evaluation tasks: use routing_expert for metric-grounded advice."
        ),
    )
    orchestrator_executor = make_executor(orchestrator_agent)

    return orchestrator_executor, orchestrator_tools


def create_orchestrator_memory_executor(llm, orchestrator_tools):
    """Create orchestrator executor with short-term conversation memory."""
    orchestrator_agent_mem = create_agent(
        model=llm,
        tools=orchestrator_tools,
        system_prompt=(
            "You are the project orchestrator for the tweet classification pipeline. "
            "Use tools to ground all answers and preserve context for follow-up questions."
        ),
    )
    return make_memory_executor(orchestrator_agent_mem)


def print_trace(result):
    """Pretty-print intermediate tool traces from agent execution."""
    steps = result.get("intermediate_steps", [])
    if not steps:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_outputs_by_id = {}
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "tool":
                call_id = getattr(msg, "tool_call_id", None)
                tool_outputs_by_id[call_id] = getattr(msg, "content", "")

        parsed_steps = []
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                tool_name = call.get("name", "unknown_tool")
                tool_input = call.get("args", {})
                call_id = call.get("id")
                observation = tool_outputs_by_id.get(call_id, "")
                parsed_steps.append((tool_name, tool_input, observation))

        if not parsed_steps:
            print("  (no tool calls in this run)")
            return

        bar = "-" * 62
        print(" ### AGENT THOUGHT TRACE ###")
        print()
        for i, (tool_name, tool_input, observation) in enumerate(parsed_steps):
            inp_str = str(tool_input)
            if len(inp_str) > 120:
                inp_str = inp_str[:120] + "-"
            obs_str = str(observation)
            if len(obs_str) > 400:
                obs_str = obs_str[:400] + "[-truncated]"

            print(f"  Step {i + 1}")
            print(bar)
            print("  Thought: (tool call captured from agent messages)")
            print(f"  Tool used: {tool_name}  - (Tool's input): ({inp_str})")
            print(f"         -  Result: {obs_str}")
            if i < len(parsed_steps) - 1:
                print()
        return

    bar = "-" * 62
    print(" ### AGENT THOUGHT TRACE ###")
    print()
    for i, (action, obs) in enumerate(steps):
        inp = action.tool_input
        if isinstance(inp, dict):
            inp_str = ", ".join(f"{k}={repr(v)}" for k, v in inp.items())
        else:
            inp_str = str(inp)
        if len(inp_str) > 120:
            inp_str = inp_str[:120] + "-"

        obs_str = str(obs)
        if len(obs_str) > 400:
            obs_str = obs_str[:400] + "[-truncated]"

        print(f"  Step {i + 1}")
        print(bar)
        print(f"  Thought: {action.log.strip()}")
        print(f"  Tool used: {action.tool}  - (Tool's input): ({inp_str})")
        print(f"         -  Result: {obs_str}")
        if i < len(steps) - 1:
            print()
