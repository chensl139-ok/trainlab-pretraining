'use strict';
// Keep existing learning deep links; the default deployment entry is the GPU workbench.
if(!location.hash)location.replace('/gpu.html');
