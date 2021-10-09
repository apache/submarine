package org.apache.submarine.server.model.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.model.database.entities.RegisteredModelTagEntity;
import org.apache.submarine.server.model.database.mappers.RegisteredModelTagMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RegisteredModelTagService {

  private static final Logger LOG = LoggerFactory.getLogger(RegisteredModelTagService.class);

  public void insert(RegisteredModelTagEntity registeredModelTag)
          throws SubmarineRuntimeException {
    LOG.info("Registered Model Tag insert name:" + registeredModelTag.getName() + ", tag:" +
            registeredModelTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelTagMapper mapper = sqlSession.getMapper(RegisteredModelTagMapper.class);
      mapper.insert(registeredModelTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert registered model tag entity to database");
    }
  }

  public void delete(RegisteredModelTagEntity registeredModelTag)
          throws SubmarineRuntimeException {
    LOG.info("Registered Model Tag delete name:" + registeredModelTag.getName() + ", tag:" +
            registeredModelTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      RegisteredModelTagMapper mapper = sqlSession.getMapper(RegisteredModelTagMapper.class);
      mapper.delete(registeredModelTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete registered model tag from database");
    }
  }
}
