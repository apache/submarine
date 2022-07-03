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
Based on `pac4j`, we plan to support popular authentication services such as OAuth2/OpenID Connect (OIDC), LDAP, Kerberos, SAML, CAS, etc.
and use a token-based method to handle external request services and internal message communication.
In the initial version we will first integrate OAuth2/OIDC, LDAP,
And a simple login mode that does not rely on other authentication services.
There are already some PRs in the community to try to integrate some authentication services into `submarine`
( [New SSO function based on OIDC](https://github.com/apache/submarine/pull/833) 和 [Create rest api to authenticate user from LDAP](https://github.com/apache/submarine/pull/419) ),
We will try to do combines on the basis of these PRs together.

### None

When supported authentication, we will also support a way to turn off authentication and call the service directly,
so that previous versions of submarine that not support authentication can call the service.
Authentication is provided by default in submarine, but we can also turn off authentication by manually setting `submarine.auth.type` to `none`.

### Simple

Provides a simple way for authentication.
When users log in to the system, the username and password entered will be matched against the `sys_user` table within the system,
and if the form is met a `token` will be generated and returned to the frontend.
All services will need to carry the `token` in the request header to confirm the user's identity.
```
Authorization: Bearer <token>
```

### OAuth2

Supports OAuth2 as a user authentication service, requiring a jump to a third-party authentication platform for single sign-on services when logging into `submarine`.
`Submarine` requires an OAuth2 token as an authentication credential, including the refresh token.
If the logged-in user is not in `submarine`, the user data will be created automatically.

### OIDC

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

### LDAP
[TODO]

### Kerberos
[TODO]

### SAML
[TODO]

### CAS
[TODO]

### Configuration

|  Attribute   | Description  | Type | Default | Comment |
|  ----  | ----  | ---- | ---- | ---- |
| submarine.auth.type  | Supported authentication types, currently available are: none, simple, oauth2/oidc, ldap, kerberos, saml, cas | string | none | Only one authentication method can be supported at any one time |
| submarine.auth.maxAge  | Expiry time of the token | int |  | |
| submarine.auth.oauth2.client.id  | OAuth2 client id | string |  | |
| submarine.auth.oauth2.client.secret  | OAuth2 client secret| string |  | |
| submarine.auth.oauth2.client.flows  | OAuth2 flows, can be: authorizationCode, implicit, password or clientCredentials | string |  | |
| submarine.auth.oauth2.scopes  | The available scopes for the OAuth2 security scheme. A map between the scope name and a short description for it. | string |  | |
| submarine.auth.oauth2.token.uri  | OAuth2 access token uri | string |  | |
| submarine.auth.oauth2.refresh.uri  | OAuth2 refresh token uri | string |  | |
| submarine.auth.oauth2.authorization.uri  | OAuth2 authorization uri | string |  | |
| submarine.auth.oauth2.logout.uri  | OAuth2 logout uri | string |  | |
| submarine.auth.oidc.client.id  | OIDC client id | string |  | |
| submarine.auth.oidc.client.secret  | OIDC client Secret| string |  | |
| submarine.auth.oidc.client.scopes  | The available scopes for the OIDC security scheme. A map between the scope name and a short description for it.| string |  | |
| submarine.auth.oidc.useNonce  | Whether to use nonce during login process | string |  | |
| submarine.auth.oidc.discover.uri  | OIDC discovery uri | string |  | |
| submarine.auth.oidc.logout.uri  | OIDC logout uri | string |  | |

## Users

Describe the design of relevant user tables, user registration/modification/deletion processes,
and the processing logic associated with authenticated login
(including the mapping of attributes for automatically registered users when integrating with other authentication platforms, etc.).

(More details TODO)

## RBAC
[TODO]


