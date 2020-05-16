import { BaseEntity } from './base-entity';

export interface Project extends BaseEntity{
    name: string; 
    description: string; 
    tags: string[];
    inputTagVisibility: boolean;
    projectInputTag: string; 
    starNum: number;
    likeNum: number; 
    messageNum: number;
    permission: string;
    projectFilesList: string[];
    teamName: string;
    type: string;
    userName: string;
    visibility: string;
  }
