import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { FormGroup, NgForm } from '@angular/forms';

@Component({
  selector: 'app-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss']
})
export class ProjectComponent implements OnInit {
  newProject = false;
  existProjects = [];

  @ViewChild('inputElement', { static: false }) inputElement: ElementRef;

  constructor() { }

  //Todo : get projects data from server
  ngOnInit() {
    this.existProjects.push({
      projectName: 'projectName0', description: 'description', tags: ['12', 'Tag 2'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2', 'Tag 3'], inputTagVisibility: false, projectInputTag: ''
    });
    this.existProjects.push({
      projectName: 'projectName1', description: 'description', tags: ['Unremovable', 'Tag 2', 'Tag 3'], inputTagVisibility: false, projectInputTag: ''
    })
  }
  //Todo: Update tag in server
  handleCloseTag(project, tag){
    project.tags = project.tags.filter(itag => itag!==tag);
    console.log(project);
    console.log(tag);
  }
  //Todo update tag in server
  handleInputConfirm(project): void {
    if (project.projectInputTag && project.tags.indexOf(project.projectInputTag) === -1) {
      project.tags = [...project.tags, project.projectInputTag];
    }
    project.inputTagVisibility = false;
    project.projectInputTag = '';
  }

  showInput(project): void {
    project.inputTagVisibility = true;
    setTimeout(() => {
      this.inputElement.nativeElement.focus();
    }, 10);
  }


}
