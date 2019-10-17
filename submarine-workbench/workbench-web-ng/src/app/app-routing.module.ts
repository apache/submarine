import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AuthGuard } from '@submarine/core';

const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: '/manager/user'
  },
  {
    path: 'manager',
    canActivate: [AuthGuard],
    loadChildren: () => import('./pages/manager/manager.module').then(m => m.ManagerModule)
  },
  {
    path: 'user',
    loadChildren: () => import('./pages/user/user.module').then(m => m.UserModule)
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
