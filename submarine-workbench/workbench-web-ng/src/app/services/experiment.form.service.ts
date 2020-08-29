/*!
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

import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { ModalProps } from '@submarine/interfaces/modal-props';

@Injectable()
export class ExperimentFormService {
  // Subject(observable source)
  private stepServiceSource = new Subject<number>();
  private fetchListServiceSource = new Subject<boolean>();
  private btnStatusServiceSource = new Subject<boolean>();
  private modalPropsServiceSource = new Subject<ModalProps>();

  // Observable streams
  stepService = this.stepServiceSource.asObservable();
  fetchListService = this.fetchListServiceSource.asObservable();
  btnStatusService = this.btnStatusServiceSource.asObservable();
  modalPropsService = this.modalPropsServiceSource.asObservable();

  // Event emitter
  stepChange(step: number) {
    this.stepServiceSource.next(step);
  }
  btnStatusChange(status: boolean) {
    this.btnStatusServiceSource.next(status);
  }
  modalPropsChange(props: ModalProps) {
    this.modalPropsServiceSource.next(props);
  }
  fetchList() {
    this.fetchListServiceSource.next(true);
  }
}
