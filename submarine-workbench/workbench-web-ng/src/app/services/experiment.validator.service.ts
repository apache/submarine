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

import { FormGroup, ValidatorFn, ValidationErrors } from '@angular/forms';
import { Injectable } from '@angular/core';
import { ExperimentModule } from '@submarine/pages/workbench/experiment/experiment.module';

/**
 * The validator for env key/value pair
 * @param envGroup A FormGroup resides in `envs` FromArray in createExperiment
 */
@Injectable({
  providedIn: ExperimentModule
})
export class ExperimentFormService {
  envValidator: ValidatorFn = (envGroup: FormGroup): ValidationErrors | null => {
    const key = envGroup.get('key');
    const keyValue = envGroup.get('value');
    return (key.value && keyValue.value) || (!key.value && !keyValue.value) ? null : { envMissing: 'Missing key or value' };
  };

  specValidator: ValidatorFn = (specGroup: FormGroup): ValidationErrors | null => {
    
  }
}
