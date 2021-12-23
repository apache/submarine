package org.apache.submarine.server.internal;

import javax.ws.rs.core.Response.Status;

import org.apache.submarine.commons.utils.exception.SubmarineRuntimeException;
import org.apache.submarine.server.api.common.CustomResourceType;
import org.apache.submarine.server.api.notebook.Notebook;
import org.apache.submarine.server.experiment.database.entity.ExperimentEntity;
import org.apache.submarine.server.experiment.database.service.ExperimentService;
import org.apache.submarine.server.notebook.database.service.NotebookService;

public class InternalServiceManager {
  private static volatile InternalServiceManager internalServiceManager;
    
  private final ExperimentService experimentService = new ExperimentService();
  private final NotebookService notebookService = new NotebookService(); 
    
  public static InternalServiceManager getInstance() {
    if (internalServiceManager == null) {
      internalServiceManager = new InternalServiceManager();
    }
    return internalServiceManager;
  }
    
  public void updateCRStatus(CustomResourceType crType, String resourceId, String status) {
    if (crType.equals(CustomResourceType.Notebook)) {
      updateNotebookStatus(resourceId, status);
    } else if (crType.equals(CustomResourceType.TFJob) || crType.equals(CustomResourceType.PYTORCHJob)) {
      updateExperimentStatus(resourceId, status);
    }
  }
    
  private void updateExperimentStatus(String resourceId, String status) {
    ExperimentEntity experimentEntity = experimentService.select(resourceId);
    if (experimentEntity == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
        String.format("cannot find experiment with id:%s", resourceId));
    }
    experimentEntity.setExperimentStatus(status);
    experimentService.update(experimentEntity);
  }
    
  private void updateNotebookStatus(String resourceId, String status) {
    Notebook notebook = notebookService.select(resourceId);
    if (notebook == null) {
      throw new SubmarineRuntimeException(Status.NOT_FOUND.getStatusCode(),
        String.format("cannot find notebook with id:%s", resourceId));
    }
    notebook.setStatus(status);
    notebookService.update(notebook);
  }
}
