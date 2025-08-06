import asyncio
import logging
import os
import sys
import traceback
from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from beeai_framework.agents.experimental.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.think import ThinkTool

from base_agent import BaseAgent, TInputSchema, TOutputSchema
from observability import setup_observability
from tools.commands import RunShellCommandTool
from tools.patch_validator import PatchValidatorTool
from utils import mcp_tools, redis_client

logger = logging.getLogger(__name__)


class InputSchema(BaseModel):
    task_id: str = Field(description="OpenScanHub Task ID")
    srpm_name: str = Field(description="SRPM Name")


class Resolution(Enum):
    PACKAGER_REVIEW_NEEDED = "packager-review-needed"
    NO_ACTION = "no-action"
    ERROR = "error"


class PackagerReviewNeededData(BaseModel):
    findings: str = Field(description="Summary of the investigation")
    additional_info_needed: str = Field(description="Summary of missing information")
    task_id: str = Field(description="Jira issue identifier")


class NoActionData(BaseModel):
    reasoning: str = Field(description="Reason why the issue is intentionally non-actionable")
    task_id: str = Field(description="Jira issue identifier")


class ErrorData(BaseModel):
    details: str = Field(description="Specific details about an error")
    task_id: str = Field(description="Jira issue identifier")


class OutputSchema(BaseModel):
    resolution: Resolution = Field(description="SAST reviewer resolution")
    data: Union[PackagerReviewNeededData, NoActionData, ErrorData] = Field(
         description="Associated data"
     )


class SASTReviewerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            llm=ChatModel.from_name(os.getenv("CHAT_MODEL")),
            tools=[ThinkTool(), RunShellCommandTool(), PatchValidatorTool()],
            memory=UnconstrainedMemory(),
            requirements=[
                ConditionalRequirement(ThinkTool, force_after=Tool, consecutive_allowed=False),
                ConditionalRequirement(RunShellCommandTool),
                ConditionalRequirement(PatchValidatorTool),
            ],
            middlewares=[GlobalTrajectoryMiddleware(pretty=True)],
        )

    @property
    def input_schema(self) -> type[TInputSchema]:
        return InputSchema

    @property
    def output_schema(self) -> type[TOutputSchema]:
        return OutputSchema

    @property
    def prompt(self) -> str:
        return """
          You are an experienced package maintainer for Red Hat Enterprise Linux with deep understanding of the upstream project you are going to analyze.
          You are running next steps in the latest stable version of Fedora.

          Goal: Analyze the given task to determine the correct course of action.

          **Initial Analysis Steps**

          1. Download and unapack the source RPM: 
             * Create a temporary directory and enter it.
             * Download {{ srpm_name }} from `https://openscanhub.fedoraproject.org/task/{{ task_id }}/log/{{ srpm_name }}?format=raw` with `curl -O`
             * If previous step fails, make it clear in the final report that SRPM could not be downloaded.
             * Remove rpmbuild directory by running `rm -rf ~/rpmbuild/`.
             * Install the SRPM by running `rpm -ivh {{ SRPM_NAME }}`.
             * Enter the directory which contains the spec file by running `cd ~/rpmbuild/SPECS/`.
             * Extract the source tarball and apply all the patches by running `rpmbuild -bp --nodeps`.
             * Enter the build directory by running `cd ~/rpmbuild/BUILD/`.
             * Thoroughly analyze the code and build a mental model of the entire codebase in this directory. Be critical of your understanding.
            
          2. Download `https://openscanhub.fedoraproject.org/task/{{ task_id }}/log/added.err?format=raw` with `curl -O`.
             
          3. Stacktrace for each individual finding is followed by lines prefixed with `Error:` or `Warning:` in the `added.err` file.

          4. Map stacktrace for each finding to the files that contain the source code in current directory.

          5. Ignore all the warnings that may be caused by GCC not understanding the cleanup attribute.

          6. Ignore any warnings which are result of xcalloc() returning NULL as this wrapper over calloc() never fails.

          7. Ignore any cppcheck warnings that are not critical.

          8. Ignore any shellcheck errors that are not important.

          9. Ignore any errors or warnings reported in the code related to tests.
          
          10. Verify that the stacktrace you are analyzing is actually executed in a code path.
 
          11. Write any specific issues that may require attention in a simple text file and request the package maintainer to verify that the code paths are executed.

            

          **Decision Guidelines & Investigation Steps**

          You must decide between one of 3 actions. Follow these guidelines to make your decision:

          1. **Packager Review Needed**
             If there are critical findings in the analysis.

          2. **No Action**
             A No Action decision is appropriate for issues that are intentionally non-actionable:
             * The report is either empty or all the findings are false positives.

          3. **Error**
             An Error decision is appropriate when there are processing issues that prevent proper analysis, e.g.:
             * The task cannot be accessed

          **Output Format**

          Your output must strictly follow the format below.

          TASK_ID: {{ task_id }}
          DECISION: rebase | backport | packager-review-needed | no-action | error

          If PackagerReview Needed:
              FINDINGS: [Summarize your understanding of the bug and what you investigated]

          If Error:
              DETAILS: [ Task can not be reviewed.]

          If No Action:
              REASONING: [Provide a concise reason why the issue is intentionally non-actionable]
        """

    async def run_with_schema(self, input: TInputSchema) -> TOutputSchema:
        tools = self._tools.copy()
        try:
            return await self._run_with_schema(input)
        finally:
            self._tools = tools

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    setup_observability(os.getenv("COLLECTOR_ENDPOINT"))
    agent = SASTReviewerAgent()

    if task_id := os.getenv("TASK_ID", None):
        srpm_name = os.getenv("SRPM_NAME", None)
        logger.info("Running in direct mode with environment variable")
        input = InputSchema(task_id=task_id, srpm_name=srpm_name)
        output = await agent.run_with_schema(input)
        # logger.info(f"Direct run completed: {output.model_dump_json(indent=4)}")
        return

    class Task(BaseModel):
        metadata: dict = Field(description="Task metadata")
        attempts: int = Field(default=0, description="Number of processing attempts")

    logger.info("Starting triage agent in queue mode")
    async with redis_client(os.getenv("REDIS_URL")) as redis:
        max_retries = int(os.getenv("MAX_RETRIES", 3))
        logger.info(f"Connected to Redis, max retries set to {max_retries}")

        while True:
            logger.info("Waiting for tasks from sast_reviewer_queue (timeout: 30s)...")
            element = await redis.brpop("sast_reviewer_queue", timeout=30)
            if element is None:
                logger.info("No tasks received, continuing to wait...")
                continue

            _, payload = element
            logger.info(f"Received task from queue")

            task = Task.model_validate_json(payload)
            input = InputSchema.model_validate(task.metadata)
            logger.info(f"Processing triage for JIRA issue: {input.issue}, "
                       f"attempt: {task.attempts + 1}")

            async def retry(task, error):
                task.attempts += 1
                if task.attempts < max_retries:
                    logger.warning(f"Task failed (attempt {task.attempts}/{max_retries}), "
                                 f"re-queuing for retry: {input.issue}")
                    await redis.lpush("sast_reviewer_queue", task.model_dump_json())
                else:
                    logger.error(f"Task failed after {max_retries} attempts, "
                               f"moving to error list: {input.issue}")
                    await redis.lpush("error_list", error)

            try:
                logger.info(f"Starting triage processing for {input.issue}")
                output = await agent.run_with_schema(input)
                logger.info(f"Triage processing completed for {input.issue}, "
                          f"resolution: {output.resolution.value}")
            except Exception as e:
                error = "".join(traceback.format_exception(e))
                logger.error(f"Exception during triage processing for {input.issue}: {error}")
                await retry(
                    task, ErrorData(details=error, task_id=input.issue).model_dump_json()
                )
            else:
                if output.resolution == Resolution.PACKAGER_REVIEW_NEEDED:
                    logger.info(f"Triage resolved as PACKAGER_REVIEW_NEEDED for {input.issue}, "
                              f"adding to clarification needed queue")
                    task = Task(metadata=output.data.model_dump())
                    await redis.lpush("clarification_needed_queue", task.model_dump_json())
                elif output.resolution == Resolution.NO_ACTION:
                    logger.info(f"Triage resolved as NO_ACTION for {input.issue}, "
                              f"adding to no action list")
                    await redis.lpush("no_action_list", output.data.model_dump_json())
                elif output.resolution == Resolution.ERROR:
                    logger.warning(f"Triage resolved as ERROR for {input.issue}, retrying")
                    await retry(task, output.data.model_dump_json())

# How to use this agent:
# make start
# podman exec -it beeai_sast-reviewer-agent_1 /bin/sh
# Examples of successful identification of bugs:
# env TASK_ID=44559 SRPM_NAME=crun-1.20-1.20250320110221641794.pr1695.62.g35863026.src.rpm python3 agents/sast_reviewer_agent.py
# env TASK_ID=49857 SRPM_NAME=openscap-1.4.3-0.20250414161233748610.pr2220.4.gc146e8d17.src.rpm python3 agents/sast_reviewer_agent.py 
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
