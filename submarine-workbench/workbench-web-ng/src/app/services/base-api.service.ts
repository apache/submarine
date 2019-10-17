import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class BaseApiService {
  baseApi: string;

  getPort() {
    let port = Number(location.port);
    if (!port) {
      port = 80;
      if (location.protocol === 'https:') {
        port = 443;
      }
    }
    return port;
  }

  getBase() {
    return `${location.protocol}//${location.hostname}:${this.getPort()}`;
  }

  getRestApiBase() {
    if (!this.baseApi) {
      this.baseApi = this.skipTrailingSlash(this.getBase()) + '/api';
    }

    return this.baseApi;
  }

  getRestApi(str: string): string {
    return `${this.getRestApiBase()}${str}`;
  }

  private skipTrailingSlash(path) {
    return path.replace(/\/$/, '');
  }
}
