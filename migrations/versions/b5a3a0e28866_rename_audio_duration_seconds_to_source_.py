"""Rename audio_duration_seconds to source_duration

Revision ID: b5a3a0e28866
Revises: 2e011b215dd0
Create Date: 2026-02-07 15:15:12.560052

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'b5a3a0e28866'
down_revision = '2e011b215dd0'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.alter_column('audio_duration_seconds', new_column_name='source_duration')


def downgrade():
    with op.batch_alter_table('transcript', schema=None) as batch_op:
        batch_op.alter_column('source_duration', new_column_name='audio_duration_seconds')
