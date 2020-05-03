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

/*
 * The idea an part of the orignal code is adpoted from Apache Spark project
 * We should obey the same Apache License 2.0 too.
 */

grammar SubmarineSqlBase;

singleStatement
    : statement EOF
    ;

statement
    : CREATE ROLE identifier                                           #createRole
    | DROP ROLE identifier                                             #dropRole
    | SHOW ROLES                                                       #showRoles
    ;

identifier
    : IDENTIFIER                                                       #unquotedIdentifier
    | quotedIdentifier                                                 #quotedIdentifierAlternative
    | nonReserved                                                      #unquotedIdentifier
    ;

quotedIdentifier
    : BACKQUOTED_IDENTIFIER
    ;

nonReserved
    : ALL
    | ALTER
    | CREATE
    | DELETE
    | DELETE
    | DROP
    | INSERT
    | PRIVILEGES
    | READ
    | ROLE
    | ROLES
    | SELECT
    | SHOW
    | UPDATE
    | USE
    | WRITE
    ;

//============================
// Start of the keywords list
//============================
ALL: 'ALL';
ALTER: 'ALTER';
CREATE: 'CREATE';
DELETE: 'DELETE';
DROP: 'DROP';
GRANT: 'GRANT';
INSERT: 'INSERT';
PRIVILEGES: 'PRIVILEGES';
READ: 'READ';
ROLE: 'ROLE';
ROLES: 'ROLES';
SELECT: 'SELECT';
SHOW: 'SHOW';
UPDATE: 'UPDATE';
USE: 'USE';
WRITE: 'WRITE';


BACKQUOTED_IDENTIFIER
    : '`' ( ~'`' | '``' )* '`'
    ;

IDENTIFIER
    : (LETTER | DIGIT | '_')+
    ;

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [A-Z]
    ;

WS  : [ \r\n\t]+ -> channel(HIDDEN)
    ;

// Catch-all for anything we can't recognize.
// We use this to be able to ignore and recover all the text
// when splitting statements with DelimiterLexer
UNRECOGNIZED
    : .
    ;
