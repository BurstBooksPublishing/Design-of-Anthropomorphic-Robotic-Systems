import importlib
import json
from typing import Dict, Any, Union

def load_descriptor(filepath: str) -> Dict[str, Any]:
    """Load JSON descriptor from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compatible(old_desc: Dict[str, Any], new_desc: Dict[str, Any], context: Dict[str, Union[float, int]]) -> bool:
    """Check if new descriptor is compatible with old one for safe upgrade."""
    # WCET must not increase to maintain real-time guarantees
    if new_desc['wcet_us'] > old_desc['wcet_us']:
        return False
    
    # Residual must stay within acceptable tolerance
    if new_desc.get('residual', 0.0) > context['residual_max']:
        return False
        
    return True

# Runtime upgrade sequence
try:
    old = load_descriptor('inv_dyn_old.json')
    new = load_descriptor('inv_dyn_new.json')
    context = {'residual_max': 5e-4}
    
    if compatible(old, new, context):
        # Hot-swap module loading
        mod = importlib.import_module(new['module_name'])
        
        # Validate module with fast deterministic self-test
        if mod.self_test():
            # Atomically switch controller reference (platform-specific implementation)
            controller.replace_inv_dyn(mod)
            
except (FileNotFoundError, json.JSONDecodeError, AttributeError, ImportError) as e:
    # Handle upgrade failures gracefully
    print(f"Upgrade failed: {e}")