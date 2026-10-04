PYTHON ?= .venv-train/bin/python
AUDITOR ?= pip-audit

.PHONY: test smoke check audit
test:
	$(PYTHON) -m pytest -q tests
	node --test tests/*.test.cjs
smoke:
	$(PYTHON) -m scripts.smoke_train
check: test
	node --check dist/gpu.js
	node --check dist/gpu-core.js
	node --check dist/app.js
	node --check dist/navigation.js
	$(PYTHON) -m pip check
audit:
	$(AUDITOR) -r server/requirements-api.txt
	$(AUDITOR) --path "$$($(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
