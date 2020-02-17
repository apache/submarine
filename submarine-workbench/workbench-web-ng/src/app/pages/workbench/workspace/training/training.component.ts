import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-training',
  templateUrl: './training.component.html',
  styleUrls: ['./training.component.scss']
})
export class TrainingComponent implements OnInit {
  isSpinning = true;
  constructor() { }

  categories = [
    {name: "Category1", enable:false},
    {name: "Category2", enable:false},
    {name: "Category3", enable:false},
    {name: "Category4", enable:false},
    {name: "Category5", enable:false},
    {name: "Category6", enable:false},
    {name: "Category7", enable:false}
  ];
  ownProcess = false;
  tagValue = ['a10', 'c12', 'tag'];
  userSelectedValue = 'noLimit';
  ratingSelectedValue = 'noLimit'
  activeUsers = ["John", "Jason"];
  ratings = ["Execellent", "Good", "Moderate"];
  
  ngOnInit() {
    
  }

  performChange(){
    console.log('cool')
  }

}
