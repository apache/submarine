import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ComponentsModule } from '@submarine/components/components.module';
import { NgZorroAntdModule } from 'ng-zorro-antd';


import { ProjectComponent } from './project/project.component';
import { ReleaseComponent } from './release/release.component';
import { TrainingComponent } from './training/training.component';
import { TeamComponent } from './team/team.component';
import { SharedComponent } from './shared/shared.component';
import { FormsModule } from '@angular/forms';
import { NewProjectPageComponent } from './project/new-project-page/new-project-page.component';

@NgModule({
    declarations: [
      ProjectComponent,
      ReleaseComponent,
      TrainingComponent,
      TeamComponent,
      SharedComponent,
      NewProjectPageComponent
    ],
    imports: [
      CommonModule,
      ComponentsModule,
      NgZorroAntdModule,
      FormsModule
    ],
    exports: [
      ProjectComponent,
      ReleaseComponent,
      TrainingComponent,
      TeamComponent,
      SharedComponent,
      NewProjectPageComponent
    ]
  })
  export class WorkspaceModule {
  }
  