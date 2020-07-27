import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'submarine-environment',
  templateUrl: './environment.component.html',
  styleUrls: ['./environment.component.scss']
})
export class EnvironmentComponent implements OnInit {
  constructor() {}
  environmentList = [
    {
      environmentName: 'my-submarine-env',
      environmentId: 'environment_1586156073228_0001',
      dockerImage: 'continuumio/anaconda3',
      kernelName: 'team_default_python_3.7',
      kernelChannels: 'defaults',
      kernelDependencies: ['_ipyw_jlab_nb_ext_conf=0.1.0=py37_0', 'alabaster=0.7.12=py37_0']
    },
    {
      environmentName: 'my-submarine-env-2',
      environmentId: 'environment_1586156073228_0002',
      dockerImage: 'continuumio/miniconda',
      kernelName: 'team_default_python_3.8',
      kernelChannels: 'defaults',
      kernelDependencies: ['_ipyw_jlab_nb_ext_conf=0.1.0=py38_0']
    }
  ];

  ngOnInit() {}
}
