import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { ReactiveFormsModule } from '@angular/forms';
import { NgZorroAntdModule } from 'ng-zorro-antd';
import { LoginComponent } from './login/login.component';
import { RegisterComponent } from './register/register.component';
import { UserRoutingModule } from './user-routing.module';
import { UserComponent } from './user.component';

@NgModule({
  declarations: [RegisterComponent, UserComponent, LoginComponent],
  imports: [CommonModule, UserRoutingModule, ReactiveFormsModule, NgZorroAntdModule]
})
export class UserModule {}
