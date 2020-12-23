/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

 import { Subscription, Subject, interval, timer } from "rxjs";

 export interface BackoffConfig {
     retries?: number;
     interval?: number;
     maxInterval?: number;
 }

 const defaultConfig: BackoffConfig = {
     retries: 1,
     interval: 1000,
     maxInterval: 16000
 }

 export class ExponentialBackoff {
     private retries: number;
     private interval: number;
     private maxInterval: number;

     private scheduler: Subscription;
     private poller: Subject<number>;
     private n: number;
     
     private remainingTries: number;
     private currInterval: number;

     constructor(config: BackoffConfig = defaultConfig) {
         const conf = { ...defaultConfig, ...config };

         this.retries = conf.retries;
         this.interval = conf.interval;
         this.maxInterval = conf.maxInterval;

         this.poller = new Subject<number>();

         this.n = 0;
         this.remainingTries = this.retries + 1;
         this.currInterval = this.interval;
     }

     public start() {
         if(this.scheduler) {
             this.scheduler.unsubscribe();
         }

         this.scheduler = timer(0, this.interval).subscribe(() => {
             this.iterate();
         });

         return this.poller;
     }

     private iterate() {
         this.n++;
         this.poller.next(this.n);

         this.scheduler.unsubscribe();
        this.remainingTries--;
        if (this.remainingTries === 0) {
            this.remainingTries = this.retries;
            this.currInterval = Math.min(this.currInterval * 2, this.maxInterval);
        }

        this.scheduler = interval(this.currInterval).subscribe(() => {
            this.iterate();
        });
    }

        public reset() {
            this.n = 0;
            this.currInterval = this.interval;
            this.remainingTries = this.retries + 1;

        this.start();
        }

        public stop() {
            if (this.scheduler) {
                this.scheduler.unsubscribe();
            }
        }

        public getPoller() {
            return this.poller;
        }
 }