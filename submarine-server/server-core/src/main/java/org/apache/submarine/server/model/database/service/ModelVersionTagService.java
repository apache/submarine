package org.apache.submarine.server.model.database.service;

import org.apache.ibatis.session.SqlSession;
import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.database.utils.MyBatisUtil;
import org.apache.submarine.server.model.database.entities.ModelVersionTagEntity;
import org.apache.submarine.server.model.database.mappers.ModelVersionTagMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ModelVersionTagService {

  private static final Logger LOG = LoggerFactory.getLogger(ModelVersionTagService.class);

  public void insert(ModelVersionTagEntity modelVersionTag)
          throws SubmarineRuntimeException {
    LOG.info("Model Version Tag insert name:" + modelVersionTag.getName() + ", tag:" +
            modelVersionTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionTagMapper mapper = sqlSession.getMapper(ModelVersionTagMapper.class);
      mapper.insert(modelVersionTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to insert model version tag entity to database");
    }
  }

  public void delete(ModelVersionTagEntity modelVersionTag)
          throws SubmarineRuntimeException {
    LOG.info("Model Version Tag delete name:" + modelVersionTag.getName() + ", tag:" +
            modelVersionTag.getTag());
    try (SqlSession sqlSession = MyBatisUtil.getSqlSession()) {
      ModelVersionTagMapper mapper = sqlSession.getMapper(ModelVersionTagMapper.class);
      mapper.delete(modelVersionTag);
      sqlSession.commit();
    } catch (Exception e) {
      LOG.error(e.getMessage(), e);
      throw new SubmarineRuntimeException("Unable to delete model version tag entity to database");
    }
  }
}
