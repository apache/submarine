/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http: //www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import { Component, OnInit } from '@angular/core';
import { ExperimentSpec } from '@submarine/interfaces/experiment-spec';
import { ModalProps } from '@submarine/interfaces/modal-props';
import { ExperimentFormService } from '@submarine/services/experiment.form.service';

@Component({
  selector: 'submarine-experiment-form',
  templateUrl: './experiment-form.component.html',
  styleUrls: ['./experiment-form.component.scss'],
})
export class ExperimentFormComponent implements OnInit {
  // About form management
  modalProps: ModalProps = {
    okText: 'Next step',
    isVisible: false,
    currentStep: 0,
    formType: null,
  };
  nextBtnDisable: boolean = true;

  // About update and clone
  mode: 'create' | 'update' | 'clone' = 'create';
  targetId: string = null;
  targetSpec: ExperimentSpec = null;

  constructor(private experimentFormService: ExperimentFormService) {}

  ngOnInit() {
    this.experimentFormService.btnStatusService.subscribe((status) => {
      this.nextBtnDisable = status;
    });
    this.experimentFormService.modalPropsService.subscribe((props) => {
      this.modalProps = { ...this.modalProps, ...props };
    });
  }

  initModal(
    initMode: 'create' | 'update' | 'clone',
    initFormType = null,
    id: string = null,
    spec: ExperimentSpec = null
  ) {
    this.mode = initMode;
    this.modalProps.isVisible = true;
    this.modalProps.formType = initFormType;

    if (initMode === 'update' || initMode === 'clone') {
      // Keep id for later request
      this.targetId = id;
      this.targetSpec = spec;
    }
  }

  closeModal() {
    this.experimentFormService.modalPropsClear();
  }

  proceedForm() {
    this.experimentFormService.stepChange(1);
  }

  prevForm() {
    this.experimentFormService.stepChange(-1);
  }
}
