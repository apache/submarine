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
