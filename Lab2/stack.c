/*
 * stack.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

pthread_mutex_t stacklock = PTHREAD_MUTEX_INITIALIZER;

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// This test fails if the task is not allocated or if the allocation failed
	
    //assert(stack->head != NULL);
    
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_t* stack, struct allocation_stack* alloc_stack, int new_data)
{
    if(alloc_stack->head == NULL)
        return 0;
#if NON_BLOCKING == 0
  // Implement a lock_based stack
    pthread_mutex_lock(&stacklock);
    element_t* old_head = stack->head;
    element_t* alloc_element = alloc_stack->head;
    
    alloc_stack->head = alloc_element->next;
   
    alloc_element->next = old_head;
    stack->head = alloc_element;
    alloc_element->data = new_data;
    pthread_mutex_unlock(&stacklock);
    
#elif NON_BLOCKING == 1
    // Implement a hardware CAS-based stack
    element_t* old_head = stack->head;
    element_t* new_head = alloc_stack->head;
    element_t* new_alloc_head = new_head->next;
    
    cas(&(stack->head), old_head, new_head);
    
    stack->head->next = old_head;
    alloc_stack->head = new_alloc_head;
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  //stack_check((stack_t*)1);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_t* stack, struct allocation_stack* alloc_stack)
{
    if(!stack->head)
        return 0;
#if NON_BLOCKING == 0
  // Implement a lock_based stack
    pthread_mutex_lock(&stacklock);
    element_t* old_head = stack->head;
    element_t* old_alloc_head = alloc_stack->head;
    
    stack->head = old_head->next;
    
    alloc_stack->head = old_head;
    alloc_stack->head->next = old_alloc_head;
    
    pthread_mutex_unlock(&stacklock);
#elif NON_BLOCKING == 1
    // Implement a harware CAS-based stack
    element_t* old_head = stack->head;
    element_t* new_head = stack->head->next;
    element_t* old_alloc_head = alloc_stack->head;
    
    cas(&(stack->head), old_head, new_head);
    
    alloc_stack->head = old_head;
    alloc_stack->head->next = old_alloc_head;
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

