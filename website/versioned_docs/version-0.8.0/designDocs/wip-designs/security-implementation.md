---
title: Security Implementation
---

<!--
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
-->


## Handle User's Credential

Users credential includes Kerberoes Keytabs, Docker registry credentials, Github ssh-keys, etc.

User's credential must be stored securitely, for example, via KeyCloak or K8s Secrets.

(More details TODO)

## Authentication

We use [pac4j](https://www.pac4j.org/) as the secure authentication component of `submarine-server`.
Based on `pac4j`, we plan to support popular authentication services such as OAuth2/OpenID Connect (OIDC), LDAP, SAML, CAS, etc.
and use a token-based method to handle external request services and internal message communication.
In the initial version we will first integrate OAuth2/OIDC, LDAP,
and a simple login mode that does not rely on other authentication services.
There are already some PRs in the community to try to integrate some authentication services into `submarine`
( [New SSO function based on OIDC](https://github.com/apache/submarine/pull/833) and [Create rest api to authenticate user from LDAP](https://github.com/apache/submarine/pull/419) ),
We will try to do combines on the basis of these PRs together.

### Supported authentication types
#### None

When supported authentication, we will also support a way to turn off authentication and call the service directly,
so that previous versions of submarine that not support authentication can call the service.
Authentication is provided by default in submarine, but we can also turn off authentication by manually setting `submarine.auth.type` to `none`.

#### Simple

Provides a simple way for authentication.
When users log in to the system, the username and password entered will be matched against the `sys_user` table within the system,
and if the form is met a `token` will be generated and returned to the frontend.
All services will need to carry the `token` in the request header to confirm the user's identity.
```
Authorization: Bearer <token>
```

#### OAuth2

Supports OAuth2 as a user authentication service, requiring a jump to a third-party authentication platform for single sign-on services when logging into `submarine`.
`Submarine` requires an OAuth2 token as an authentication credential, including the refresh token.
If the logged-in user is not in `submarine`, the user data will be created automatically.

#### OIDC

OIDC is similar to OAuth2, except that `submarine.auth.oidc.discover.uri` is required to support [OpenID Connect Discovery](https://openid.net/specs/openid-connect-discovery-1_0.html),
where an OpenID server publishes its metadata at a well-known URL, typically
```
https://server.com/.well-known/openid-configuration
```

This URL returns a JSON listing of the OpenID/OAuth endpoints, supported scopes and claims, public keys used to sign the tokens, and other details.
The `pac4j` can use this information to construct a request to the OpenID server.
The field names and values are defined in the OpenID Connect Discovery Specification. Here is an example of data returned:

```json
{
    "issuer": "https://example.com/",
    "authorization_endpoint": "https://example.com/authorize",
    "token_endpoint": "https://example.com/token",
    "userinfo_endpoint": "https://example.com/userinfo",
    "jwks_uri": "https://example.com/.well-known/jwks.json",
    "scopes_supported": [
        "pets_read",
        "pets_write",
        "admin"
    ],
    "response_types_supported": [
        "code",
        "id_token",
        "token id_token"
    ],
    "token_endpoint_auth_methods_supported": [
      "client_secret_basic"
    ],
    ...
}
```

#### LDAP
[TODO]

#### SAML
[TODO]

#### CAS
[TODO]

### Configuration

| Attribute                               | Description                                                                                                       | Type    | Default | Comment                                                                                             |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|---------|---------|-----------------------------------------------------------------------------------------------------|
| submarine.auth.type                     | Supported authentication types, currently available are: none, simple, oauth2/oidc, ldap, kerberos, saml, cas     | string  | none    | Only one authentication method can be supported at any one time                                     |
| submarine.auth.token.maxAge             | Expiry time of the token (minute)                                                                                 | int     | 1 day   |                                                                                                     |
| submarine.auth.refreshToken.maxAge      | Expiry time of the refresh token (minute)                                                                         | int     | 1 hour  |                                                                                                     |
| submarine.cookie.http.only              | HttpOnly Cookie                                                                                                   | boolean | false   |                                                                                                     |
| submarine.cookie.secure                 | Secure Cookie                                                                                                     | boolean | false   |                                                                                                     |
| submarine.cookie.samesite               | SameSite Cookie, can be Lax, Strict, None(or empty)                                                               | string  |         | https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite                       |
| submarine.auth.oauth2.client.id         | OAuth2 client id                                                                                                  | string  |         |                                                                                                     |
| submarine.auth.oauth2.client.secret     | OAuth2 client secret                                                                                              | string  |         |                                                                                                     |
| submarine.auth.oauth2.client.flows      | OAuth2 flows, can be: authorizationCode, implicit, password or clientCredentials                                  | string  |         |                                                                                                     |
| submarine.auth.oauth2.scopes            | The available scopes for the OAuth2 security scheme. A map between the scope name and a short description for it. | string  |         |                                                                                                     |
| submarine.auth.oauth2.token.uri         | OAuth2 access token uri                                                                                           | string  |         |                                                                                                     |
| submarine.auth.oauth2.refresh.uri       | OAuth2 refresh token uri                                                                                          | string  |         |                                                                                                     |
| submarine.auth.oauth2.authorization.uri | OAuth2 authorization uri                                                                                          | string  |         |                                                                                                     |
| submarine.auth.oauth2.logout.uri        | OAuth2 logout uri                                                                                                 | string  |         |                                                                                                     |
| submarine.auth.oidc.client.id           | OIDC client id                                                                                                    | string  |         |                                                                                                     |
| submarine.auth.oidc.client.secret       | OIDC client Secret                                                                                                | string  |         |                                                                                                     |
| submarine.auth.oidc.discover.uri        | OIDC discovery uri                                                                                                | string  |         |                                                                                                     |
| submarine.auth.ladp.provider.uri        | LDAP provider uri                                                                                                 | string  |         |                                                                                                     |
| submarine.auth.ladp.baseDn              | LDAP base DN                                                                                                      | string  |         | base DN is the base LDAP distinguished name for your LDAP server. For example, ou=dev,dc=xyz,dc=com |
| submarine.auth.ladp.domain              | LDAP AD domain                                                                                                    | string  |         | AD domain is the domain name of the AD server. For example, corp.domain.com                         |

### Design and implementation

We use `javax.servlet.Filter` in the server to determine if authentication information exists for a user.
The `Filter` is implemented for each authentication type and is configured according to the implementation of the type specified by `pac4j`.
Also, a `SecurityFactory` class is provided that instantiates the specified `Filter` class into Jetty's filter based on `submarine.auth.type`.

Except in the case of `submarine.auth.type` being `none`, and some APIs necessary for authentication (login requests, etc.), we will require the token to be included in the header.
The token is generated and verified based on `pac4j` and processed inside the `Filter` class, incorrect token or no token will return a 401 HTTP code.

When a token expires, it can be regenerated by calling the refresh token method. The default token expiry time is now set to 1 day (by modifying `submarine.auth.token.maxAge`) and the refresh token expiry time is 1 hour.

### Users

Describe the design of relevant user tables, user registration/modification/deletion processes,
and the processing logic associated with authenticated login
(including the mapping of attributes for automatically registered users when integrating with other authentication platforms, etc.).

We use `sys_user` table to store user information for submarines.
When `submarine.auth.type` is `simple`, the user's login operation will match `user_name` and `password` (encrypted) in `sys_user`. Only when the user name and password match will the login succeed.
When `submarine.auth.type` is `ldap`, the user's login will operation request the LDAP and verify that the username and password are correct. A new record will be added to the `sys_user` table if the logged-in user does not exist.
When logging in using other third-party authentication (OAuth2/OpenID Connect (OIDC), SAML, CAS etc.), the login page will automatically jump to the third-party service and revert back to the submarine after a successful login. A new record will be added to the `sys_user` table if the logged-in user does not exist.

#### Department
[TODO]

#### Role
[TODO]

### RBAC
[TODO]


