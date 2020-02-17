import { Component, OnInit, ViewChild, Output, EventEmitter, Input } from '@angular/core';
import { NgForm } from '@angular/forms';


@Component({
  selector: 'app-new-project-page',
  templateUrl: './new-project-page.component.html',
  styleUrls: ['./new-project-page.component.scss']
})
export class NewProjectPageComponent implements OnInit {
  @Output() closeProjectPage = new EventEmitter<boolean>();
  @ViewChild('f', { static: true }) signupForm: NgForm;
  //Todo: get team from API
  teams = ['ciil'];
  
  current = 0;
  
  newProjectContent = { projectName: 'projectName', description: 'description', visibility: 'Private', team: '' ,permission: 'view', dataSet: []};
  

  constructor() { }

  ngOnInit() {
  }


  clearProject(){
    this.closeProjectPage.emit(true);
  }

  pre(): void {
    this.current -= 1;
  }

  next(): void {
    this.current += 1;
  }

  //Todo : Add the new project
  done(): void{
    console.log(this.newProjectContent);
    this.clearProject();
  }

  //Todo : open in notebook
  openNotebook() {
    ;
  }
}
