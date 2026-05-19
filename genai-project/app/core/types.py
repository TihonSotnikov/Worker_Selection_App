import asyncio
import inspect

from collections import defaultdict


class AdvancedLock(asyncio.Lock):
    """
    Асинхронный лок с поддержкой хуков. \\
    Может выполнять обработчики захвата, освобождения
    и пользовательских событий:
    ```
    lock = AdvancedLock()

    @lock.on("acquire")
    def on_acquire():
        print("Лок захвачен.")
    
    @lock.on("your_event")
    def on_event(txt):
        print(f'Произошло непредвиденное: {txt}')
    
    with lock as l:
        l.emit("your_event", 67)
    
    #> Лок захвачен.
    #> Произошло непредвиденное: 67
    ```

    Parameters
    ----------
    asyncio : _type_
        _description_
    """

    def __init__(self):
        super().__init__()
        self._hooks = defaultdict(list)

    def on(self, event: str):
        """
        Универсальный декоратор регистрации хука. Стандартные события:
        - @lock.on('acquire')
        - @lock.on('release')
        """
        def decorator(func):
            self._hooks[event].append(func)
            return func
        return decorator

    def on_acquire(self):
        """Алиас для @lock.on('acquire')"""
        return self.on("acquire")

    def on_release(self):
        """Алиас для @lock.on('release')"""
        return self.on("release")
    
    async def emit(self, event: str, *args, **kwargs):
        """Метод для генерации (вызова) события с передачей аргументов"""
        if event not in self._hooks:
            return

        for hook in self._hooks[event]:
            if inspect.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)

    async def acquire(self):
        await super().acquire()
        try:
            await self.emit("acquire")
        except Exception:
            super().release()
            raise
        return True

    def release(self):
        super().release()
        for hook in self._hooks["release"]:
            if inspect.iscoroutinefunction(hook):
                asyncio.create_task(hook())
            else:
                hook()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        super().release()
        await self.emit("release")
