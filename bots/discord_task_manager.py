import atexit
import json

import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

task_lists = {}
active_task_list = None


@bot.command(name='newlist')
async def create_task_list(ctx, name):
    global task_lists
    global active_task_list
    task_lists[name] = []
    active_task_list = name
    await ctx.send(f'Task list "{name}" created and set as active.')


@bot.command(name='active')
async def change_active_list(ctx, name=None):
    global active_task_list
    if name and name in task_lists:
        active_task_list = name
        await ctx.send(f'Active task list changed to "{name}".')
    elif not name and active_task_list:
        await ctx.send(f'Active task list is already "{active_task_list}".')
    else:
        await ctx.send('Task list not found.')


@bot.command(name='add')
async def add_task(ctx, description, task_list_name=None):
    global task_lists
    global active_task_list
    if not task_list_name:
        task_list_name = active_task_list

    if task_list_name in task_lists:
        task_index = len(task_lists[task_list_name])
        task_lists[task_list_name].append({'description': description, 'index': task_index, 'users': []})
        await ctx.send(f'Task added to "{task_list_name}" with index {task_index}.')
    else:
        await ctx.send('Task list not found.')


@bot.command(name='assign')
async def assign_to_task(ctx, task_index, username=None):
    # This function will allow a user to assign themselves or another user to a task by the task index
    global task_lists
    global active_task_list
    task_list = task_lists.get(active_task_list, None)
    if task_list and 0 <= task_index < len(task_list):
        task = task_list[task_index]
        if username is None:
            username = str(ctx.author)
        elif not ctx.author.guild_permissions.administrator:
            await ctx.send('You must have admin privileges to assign a task to someone else.')
            return
        task['users'].append(username)
        await ctx.send(f'User {username} assigned to task with index {task_index}.')
    else:
        await ctx.send('Task not found in the active task list.')


@bot.command(name='complete')
async def complete_task(ctx, task_index):
    global task_lists
    global active_task_list
    task_list = task_lists.get(active_task_list, None)
    username = str(ctx.author)
    if task_list and 0 <= task_index < len(task_list):
        task = task_list[task_index]
        if username in task['users']:
            task['completed'] = True
            await ctx.send(f'Task with index {task_index} marked as completed by {username}.')
        else:
            await ctx.send('You are not assigned to this task.')
    else:
        await ctx.send('Task not found in the active task list.')


@bot.command(name='remove')
async def remove_task(ctx, task_index):
    global task_lists
    global active_task_list
    task_list = task_lists.get(active_task_list, None)
    if task_list and 0 <= task_index < len(task_list):
        task_list.pop(task_index)
        for i, task in enumerate(task_list[task_index:]):
            task['index'] = task_index + i
        await ctx.send(f'Task with index {task_index} removed from the active task list.')
    else:
        await ctx.send('Task not found in the active task list.')


@bot.command(name='display')
async def display_task_lists(ctx):
    global task_lists
    if task_lists:
        for task_list_name, tasks in task_lists.items():
            response = f'Task List "{task_list_name}":\n'
            for task in tasks:
                users = ', '.join(task['users'])
                status = 'Completed' if task.get('completed') else 'Pending'
                response += f"  - Task {task['index']}: {task['description']} (Assigned to: {users}, Status: {status})\n"
            await ctx.send(response)
    else:
        await ctx.send('No task lists found.')


def load_task_lists():
    global task_lists
    try:
        with open('task_lists.json', 'r') as file:
            task_lists = json.load(file)
    except FileNotFoundError:
        task_lists = {}


def save_task_lists():
    with open('task_lists.json', 'w') as file:
        json.dump(task_lists, file)


def main():
    atexit.register(save_task_lists)

    load_task_lists()

    token = ''
    bot.run(token)


if __name__ == "__main__":
    main()
