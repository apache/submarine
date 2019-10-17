import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LocalStorageService {
  prefix = 'submarine';

  generateKey(key: string) {
    return `${this.prefix}_${key.toLowerCase()}`;
  }

  get<T>(key: string): T {
    try {
      return JSON.parse(localStorage.getItem(this.generateKey(key)));
    } catch (e) {
      return null;
    }
  }

  set(key: string, value: string | number | object | any[]) {
    try {
      const saveValue = JSON.stringify(value);

      window.localStorage.setItem(this.generateKey(key), saveValue);
    } catch (e) {
      // empty
    }
  }

  remove(key: string) {
    window.localStorage.removeItem(this.generateKey(key));
  }
}
