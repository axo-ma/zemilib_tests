from zemi.component import ZemiComponent


component = ZemiComponent()
try:
    component.run()
finally:
    component.close()
