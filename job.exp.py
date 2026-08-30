from zemi.component import ZemiComponent


component = ZemiComponent()
try:
    for playbook in component.playbooks:
        if playbook.enabled:
            playbook.run()
except Exception as error:
    component.report.record_failure(error)
    raise
finally:
    component.close()
