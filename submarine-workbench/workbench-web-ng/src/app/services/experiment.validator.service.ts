import { ValidatorFn, FormGroup, ValidationErrors } from '@angular/forms';
/**
 * The validator for env key/value pair
 * @param envGroup A FormGroup resides in `envs` FromArray in createExperiment
 */
export const envValidator: ValidatorFn = (envGroup: FormGroup): ValidationErrors | null => {
  const key = envGroup.get('key');
  const keyValue = envGroup.get('value');
  // return (key.dirty || keyValue.dirty) || ((!key.value && keyValue.value) || (key.value && !keyValue.value)) ? { envMissing: 'Missing key or value' } : null;
  return (key.value && keyValue.value) || (!key.value && !keyValue.value) ? null : { envMissing: 'Missing key or value' };
};
