import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { EnvironmentRoutingModule } from './environment-routing.module';
import { RouterModule } from '@angular/router';
import { EnvironmentHomeComponent } from './environment-home/environment-home.component';
import { EnvironmentService } from '@submarine/services/environment-services/environment.service';
import { EnvironmentComponent } from './environment.component';
import { EnvironmentListComponent } from './environment-home/environment-list/environment-list.component';
import { EnvironmentFormComponent } from './environment-home/environment-form/environment-form.component';
import { EnvironmentInfoComponent } from './environment-info/environment-info.component';

@NgModule({
  declarations: [
    EnvironmentComponent,
    EnvironmentHomeComponent,
    EnvironmentListComponent,
    EnvironmentFormComponent,
    EnvironmentInfoComponent,
  ],
  imports: [CommonModule, ReactiveFormsModule, FormsModule, RouterModule, NgZorroAntdModule, EnvironmentRoutingModule],
  providers: [EnvironmentService],
  exports: [EnvironmentComponent],
})
export class EnvironmentModule {}
