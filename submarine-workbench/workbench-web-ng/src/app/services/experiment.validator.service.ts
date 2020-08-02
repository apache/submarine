/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { FormGroup, ValidatorFn, ValidationErrors, FormControl, FormArray } from '@angular/forms';
import { Injectable } from '@angular/core';
import { ExperimentModule } from '@submarine/pages/workbench/experiment/experiment.module';

@Injectable({
  providedIn: ExperimentModule
})
export class ExperimentFormService {
  /**
   * The validator for env key/value pair
   * @param envGroup A FormGroup resides in `envs` FromArray in createExperiment
   */
  envValidator: ValidatorFn = (envGroup: FormGroup): ValidationErrors | null => {
    const key = envGroup.get('key');
    const keyValue = envGroup.get('value');
    return !(key.invalid || keyValue.invalid)
      ? null
      : { envMissing: 'Missing key or value' };
  };

  specValidator: ValidatorFn = (specGroup: FormGroup): ValidationErrors | null => {
    const name = specGroup.get('name');
    const replicas = specGroup.get('replicas');
    const cpus = specGroup.get('cpus');
    const memory = specGroup.get('memory');

    const allValid = !(name.invalid || replicas.invalid || cpus.invalid || memory.invalid);
    return allValid ? null : { specError: 'Invalid or missing input' };
  };

  /**
   * Validate memory input in Spec
   *
   * @param memory - The memory group in Spec, containing actual number and unit
   */
  memoryValidator: ValidatorFn = (memoryGroup: FormGroup): ValidationErrors | null => {
    // Must match number + digit ex. 512M or empty
    const memory = `${memoryGroup.get('num').value}${memoryGroup.get('unit').value}`;
    
    return /^\d+[GM]$/.test(memory)
      ? null
      : { memoryPatternError: 'Memory pattern must match number + (G or M) ex. 512M' };
  };

  /**
   * Validate name or key property
   * Name and key cannot have its duplicate, must be unique
   * @param fieldName - The field name of the form
   * @returns The actual ValidatorFn to check duplicates
   */
  nameValidatorFactory: (fieldName: string) => ValidatorFn = (fieldName) => {
    return (arr: FormArray): ValidationErrors | null => {
      const duplicateSet = new Set();
      for (let i = 0; i < arr.length; i++) {
        const nameControl = arr.controls[i].get(fieldName);
        // We don't consider empty string
        if (!nameControl.value) continue;

        if (duplicateSet.has(nameControl.value)) {
          // Found duplicates, manually set errors on FormControl level
          nameControl.setErrors({
            duplicateError: 'Duplicate key or name'
          });
        } else {
          duplicateSet.add(nameControl.value);
          if (nameControl.hasError('duplicateError')) {
            delete nameControl.errors.duplicateError;
            nameControl.updateValueAndValidity();
          }
        }
      }
      return null;
    };
  };
}
