import importlib.util
import traceback

spec = importlib.util.spec_from_file_location('bb', r'e:/EduCAT/scripts/bobcat_train.py')
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('Loaded module, has MAMLModel =', hasattr(mod, 'MAMLModel'))
except Exception as e:
    print('Exception during import:')
    traceback.print_exc()
    raise
