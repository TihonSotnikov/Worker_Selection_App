from fastapi.datastructures import State

__app_state: State | None = None

def get_app_state() -> State:
    global __app_state
    if __app_state is None:
        raise RuntimeError("App state has not been initialized yet.")
    return __app_state

def set_app_state(state: State) -> None:
    global __app_state
    __app_state = state
