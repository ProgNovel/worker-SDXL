INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },
      'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    }
}
