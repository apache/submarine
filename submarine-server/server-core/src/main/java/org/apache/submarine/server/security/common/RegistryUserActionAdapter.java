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

package org.apache.submarine.server.security.common;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.submarine.server.database.workbench.entity.SysUserEntity;
import org.apache.submarine.server.database.workbench.service.SysUserService;
import org.pac4j.core.context.JEEContext;
import org.pac4j.core.exception.http.HttpAction;
import org.pac4j.core.http.adapter.JEEHttpActionAdapter;
import org.pac4j.core.profile.CommonProfile;
import org.pac4j.core.profile.ProfileManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Date;
import java.util.Optional;

import static org.apache.submarine.server.database.workbench.service.SysUserService.DEFAULT_ADMIN_UID;
import static org.apache.submarine.server.database.workbench.service.SysUserService.DEFAULT_CREATE_USER_PASSWORD;

/**
 * Triggers automatic creation of non-existent users
 * when authenticating third party logins to the adapter at the same time
 */
public class RegistryUserActionAdapter<T extends CommonProfile, R extends JEEContext>
        extends JEEHttpActionAdapter {

  private final Logger LOG = LoggerFactory.getLogger(RegistryUserActionAdapter.class);
  private final boolean loadFromSession;
  private static final SysUserService userService = new SysUserService();

  public RegistryUserActionAdapter(boolean loadFromSession) {
    this.loadFromSession = loadFromSession;
  }

  @Override
  public T adapt(final HttpAction action, final JEEContext context) {
    super.adapt(action, context);
    // get profile
    ProfileManager<T> manager = new ProfileManager<>(context);
    Optional<T> profile = manager.get(loadFromSession);
    // every time call back, check if this user is exists
    profile.ifPresent(this::createUndefinedUser);
    return null;
  }

  /**
   * Create a user that does not exist
   */
  protected void createUndefinedUser(CommonProfile ... profiles) {
    LOG.trace("Check user if exists ...");
    for (CommonProfile profile : profiles) {
      try {
        // If the user does not exist then create
        userService.getOrCreateUser(profile.getUsername(), () -> {
          SysUserEntity entity = new SysUserEntity();
          entity.setUserName(profile.getUsername());
          entity.setRealName(profile.getDisplayName());
          entity.setPassword(DEFAULT_CREATE_USER_PASSWORD);
          entity.setEmail(profile.getEmail());
          entity.setPhone(ObjectUtils.identityToString(profile.getAttribute("phone")));
          entity.setAvatar(ObjectUtils.identityToString(profile.getPictureUrl()));
          entity.setDeleted(0);
          entity.setCreateBy(DEFAULT_ADMIN_UID);
          entity.setCreateTime(new Date());
          return entity;
        });
      } catch (Exception e) {
        LOG.error("Get error when creating user, skip ...", e);
      }
    }
  }
}
