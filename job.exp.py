from zemi import arsenal
from zemi.arsenal import ArsenalSession
from zemi.component import ZemiComponent


component = ZemiComponent()
arsenal_session = ArsenalSession(component.arsenal_config_path)
try:
    arsenal.begin(arsenal_session, stop_before_begin=True)
    component.run()
finally:
    arsenal.end(arsenal_session, stop_after_end=True)
    component.close()
