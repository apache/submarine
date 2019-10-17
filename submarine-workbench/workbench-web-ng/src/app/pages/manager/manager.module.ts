import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { ManagerRoutingModule } from './manager-routing.module';
import { ManagerComponent } from './manager.component';
import { UserComponent } from './user/user.component';

@NgModule({
  declarations: [UserComponent, ManagerComponent],
  imports: [CommonModule, ManagerRoutingModule, NgZorroAntdModule]
})
export class ManagerModule {}
