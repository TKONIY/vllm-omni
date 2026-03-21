# PR4: LeRobot gRPC API for World Model Serving

**PR Scope:** Implement LeRobot's `AsyncInference` gRPC service in vllm-omni, enabling LeRobot's `RobotClient` to connect directly to vllm-omni's DiffusionEngine without code changes.

---

## Motivation

LeRobot has the largest robotics ecosystem: 12+ robots, 6 simulation environments, standardized datasets. Its `RobotClient` uses gRPC `AsyncInference` service with **decoupled obs/action streams** — obs reception and action inference run independently, enabling:

- **Obs queue with overwrite:** New obs replaces old (maxsize=1 queue), server always infers on latest
- **No request-response coupling:** Client sends obs and requests actions independently

This is superior to the serial WebSocket pattern (OpenPI/DreamZero) for real-time control, where the serial loop wastes ~50% GPU time waiting for client.

## gRPC Service Definition

From LeRobot's `services.proto`:

```protobuf
service AsyncInference {
  rpc SendObservations(stream Observation) returns (Empty);
  rpc GetActions(Empty) returns (Actions);
  rpc SendPolicyInstructions(PolicySetup) returns (Empty);
  rpc Ready(Empty) returns (Empty);
}
```

## Implementation

**New file:** `vllm_omni/entrypoints/openai/serving_world_grpc.py`

```python
class VLLMOmniPolicyServer(services_pb2_grpc.AsyncInferenceServicer):

    def __init__(self, engine: DiffusionEngine, session_store: WorldSessionStore):
        self.engine = engine
        self.session_store = session_store
        self.obs_queue = Queue(maxsize=1)

    def Ready(self, request, context):
        """Client handshake — reset server state."""
        self.obs_queue = Queue(maxsize=1)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):
        """Client sends policy config — vllm-omni ignores (model already loaded)."""
        # Read lerobot_features for obs key mapping
        policy_specs = pickle.loads(request.data)
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):
        """Receive obs, put in queue (overwrite old)."""
        received_bytes = receive_bytes_in_chunks(request_iterator, ...)
        timed_obs = pickle.loads(received_bytes)
        if self.obs_queue.full():
            self.obs_queue.get_nowait()  # discard old
        self.obs_queue.put(timed_obs)
        return services_pb2.Empty()

    def GetActions(self, request, context):
        """Take latest obs from queue → engine.step → return actions."""
        obs = self.obs_queue.get(timeout=self.obs_queue_timeout)
        session = self.session_store.get_or_create_sync(obs.session_id)
        request = self._build_diffusion_request(obs, session)
        result = self.engine.step(request)
        self._update_session(session, result)
        action_chunk = self._convert_to_timed_actions(result, obs)
        return services_pb2.Actions(data=pickle.dumps(action_chunk))
```

## Key Design Decisions

- **`SendPolicyInstructions` ignores model loading** — vllm-omni loads model at startup via `vllm serve`. Client's `policy_type` and `pretrained_name_or_path` are ignored. Only `lerobot_features` (obs key mapping) and `actions_per_chunk` are used.
- **Session state** — same `WorldSessionStore` + `extra_args` round-trip as WebSocket serving (PR3).
- **Obs format conversion** — LeRobot uses per-joint keys (`shoulder_pan.pos`), converted to vllm-omni format in `_build_diffusion_request` using `lerobot_features` mapping.
- **Action format** — returns `list[TimedAction]` (LeRobot's format with timestamps), pickled.

## Compatibility

LeRobot `RobotClient` connects without any code change:
```bash
# LeRobot client points to vllm-omni server
python src/lerobot/async_inference/robot_client.py \
    --server_address=vllm-omni-host:8080 \
    --policy_type=dreamzero \
    --pretrained_name_or_path=GEAR-Dreams/DreamZero-DROID
```

## Dependencies

- PR3 (multi-turn stateful engine) — for session state management
- PR5 (OpenPI WebSocket serving) — shares `OmniServingWorld` business layer and `session_manager.py`
- LeRobot's `services_pb2.py` / `services_pb2_grpc.py` — vendored or generated from proto
