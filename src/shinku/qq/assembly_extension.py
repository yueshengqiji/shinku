"""Runtime-extension binding and scene prompt construction."""
from __future__ import annotations

def bind_extension(
    config,
    runtime,
    persona,
    extension,
    command_state,
    memory_service,
    self_state_provider,
    task_coordinator,
    effective_tool_host,
    *,
    extension_error,
    config_error,
):
    system_prompt = persona.text
    command_interpreter = None
    if extension is not None:
        try:
            extension.bind_runtime(
                registry=effective_tool_host.registry, runtime=runtime,
                permission_provider=command_state.permission_mode,
                enabled_provider=command_state.is_agent_enabled,
            )
            system_prompt = extension.augment_system_prompt(prompt=system_prompt, config=config)
            command_interpreter = extension.build_command_interpreter(
                state=command_state,
                runtime=runtime,
                memory_inspector=memory_service.inspect if memory_service is not None else None,
                memory_reset=memory_service.reset_all if memory_service is not None else None,
                runtime_snapshot=self_state_provider,
                task_controller=task_coordinator.control,
                config=config,
            )
        except extension_error as exc:
            raise config_error(str(exc)) from exc
    return system_prompt, command_interpreter


def build_scene_prompts(
    config,
    extension,
    *,
    scene_files,
    environ,
    persona_loader,
    persona_error,
    extension_error,
    logger,
):
    scene_prompts: dict[str, str] = {}
    for conversation_type, scene_file in scene_files(environ).items():
        try:
            scene_persona = persona_loader(scene_file)
            scene_prompt = scene_persona.text
            if extension is not None:
                scene_prompt = extension.augment_system_prompt(prompt=scene_prompt, config=config)
        except (persona_error, extension_error) as exc:
            logger.warning(
                "persona_scene_skipped conversation_type=%s path=%s reason=%s",
                conversation_type, scene_file, exc,
            )
            continue
        scene_prompts[conversation_type] = scene_prompt
    return scene_prompts
